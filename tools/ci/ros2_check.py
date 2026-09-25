#!/usr/bin/env python3
"""
Checks the ROS 2 packages of chapter 19 the way the README builds them, and
then that each node does its job on synthetic messages.

Steps, in order (select some with --steps):

  deps    rosdep install --from-paths 19_vision_ros2, as in the README
  build   colcon build --base-paths 19_vision_ros2, with no compiler warning
  test    colcon test: the ament_lint linters the packages enable
  smoke   each node fed by this script, and its output checked:
            opencv_demo     bgr8 image in, mono8 image out, same header
            transport_demo  the same through image_transport
            sync_demo       left and right with close stamps, both windows open
            pcl_demo        XYZRGB cloud in, the same points out, same header
            launch_demo     depth + colour + camera_info in, cloud out, and
                            pcl_demo consuming it: the chain of the chapter

It runs inside the image of tools/ci/Dockerfile.ros2:

  xvfb-run -a python3 tools/ci/ros2_check.py --src /src --work /work

Exit code 0 if everything passed, 1 otherwise.
"""

import argparse
import os
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_examples import Report, own_cmake_warnings, prepare, run  # noqa: E402

STEPS = ['deps', 'build', 'test', 'smoke']
PKG_DIR = '19_vision_ros2'


def ros_env(root):
    """The environment after sourcing the workspace, as a dict."""
    out = subprocess.check_output(
        ['bash', '-c', 'source install/setup.bash && env -0'], cwd=root)
    return dict(line.split('=', 1) for line in out.decode().split('\0') if '=' in line)


# --- Steps ------------------------------------------------------------------------

def step_deps(root, report, args):
    run(['rosdep', 'update'], cwd=root)
    code, out, _ = run(['apt-get', 'update'], cwd=root)
    code, out, _ = run(['rosdep', 'install', '--from-paths', PKG_DIR,
                        '--ignore-src', '-r', '-y'], cwd=root)
    report.add('deps', 'rosdep install', 'PASS' if code == 0 else 'FAIL', out if code else '')


def step_build(root, report, args):
    code, out, secs = run(['colcon', 'build', '--base-paths', PKG_DIR, '--symlink-install',
                           '--event-handlers', 'console_direct+'], cwd=root)
    if code != 0:
        report.add('build', 'colcon build', 'FAIL', out)
        return
    warnings = [l for l in out.splitlines() if re.search(r'\bwarning:', l)]
    warnings += own_cmake_warnings(out, root)
    report.add('build', 'colcon build', 'FAIL' if warnings else 'PASS',
               '\n'.join(warnings) if warnings else '%.0f s' % secs)


def step_test(root, report, args):
    run(['colcon', 'test', '--base-paths', PKG_DIR], cwd=root)
    code, out, _ = run(['colcon', 'test-result', '--verbose'], cwd=root)
    report.add('test', 'colcon test (ament_lint)', 'PASS' if code == 0 else 'FAIL',
               out if code else out.strip().splitlines()[-1] if out.strip() else '')


def step_smoke(root, report, args):
    # The smoke test needs rclpy with the workspace sourced: it runs as a child.
    env = ros_env(root)
    env['ROS_DOMAIN_ID'] = str(args.domain)
    code, out, _ = run([sys.executable, os.path.abspath(__file__), '--smoke-child'],
                       cwd=root, env=env, timeout=600)
    for line in out.splitlines():
        if line.startswith('RESULT\t'):
            _, name, status, detail = line.split('\t', 3)
            report.add('smoke', name, status, detail.replace('\\n', '\n'))
    if code not in (0, 1):
        report.add('smoke', 'smoke test process', 'FAIL', out)


# --- The smoke test proper, in the sourced environment ----------------------------

def smoke_child():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

    W, H = 64, 48

    def image(stamp, encoding, frame='camera'):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.width, msg.height, msg.encoding = W, H, encoding
        if encoding == '16UC1':
            msg.step = W * 2
            msg.data = struct.pack('<%dH' % (W * H), *([1500] * (W * H)))  # 1.5 m
        else:
            msg.step = W * 3
            msg.data = bytes((x * 4) % 256 for _ in range(H) for x in range(W) for _ in range(3))
        return msg

    def camera_info(stamp, frame='camera'):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.width, msg.height = W, H
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0] * 5
        msg.k = [50.0, 0.0, W / 2, 0.0, 50.0, H / 2, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [50.0, 0.0, W / 2, 0.0, 0.0, 50.0, H / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def cloud(stamp, n=100, frame='camera'):
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.height, msg.width = 1, n
        msg.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                      PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                      PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                      PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1)]
        msg.point_step, msg.row_step = 32, 32 * n
        msg.is_dense = True
        data = bytearray()
        for i in range(n):
            rgb = struct.unpack('<f', struct.pack('<I', 0x00ff0000 | i))[0]
            data += struct.pack('<fff4xf12x', 0.01 * i, 0.0, 1.0, rgb)
        msg.data = bytes(data)
        return msg

    class Harness(Node):
        def __init__(self):
            super().__init__('ci_smoke')
            self.received = {}

        def listen(self, topic, msg_type):
            self.received[topic] = None

            def keep(msg):
                if self.received[topic] is None:
                    self.received[topic] = msg
            self.create_subscription(msg_type, topic, keep, qos_profile_sensor_data)

        # Reliable, which every subscriber accepts: the nodes of the chapter
        # ask for best effort, depth_image_proc for reliable.
        def publisher(self, topic, msg_type):
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            return self.create_publisher(msg_type, topic, qos)

    def start(cmd):
        # The log goes to a file: a pipe nobody reads until the end would fill up
        # and block a talkative node.
        log = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        proc.log = log
        return proc

    def stop(proc):
        alive = proc.poll() is None
        if alive:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        proc.log.seek(0)
        out = proc.log.read().decode('utf-8', 'replace')
        return alive, out

    def emit(name, ok, detail=''):
        print('RESULT\t%s\t%s\t%s' % (name, 'PASS' if ok else 'FAIL',
                                     detail.replace('\n', '\\n')), flush=True)

    def windows():
        out = subprocess.run(['xwininfo', '-root', '-tree'], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL).stdout.decode('utf-8', 'replace')
        return out

    def scenario(name, cmds, feed, outputs, check, timeout=30.0):
        """Starts cmds, calls feed(stamp) at 10 Hz until every output topic has a
        message or the time runs out, then stops the nodes and checks."""
        rclpy.init()
        node = Harness()
        for topic, msg_type in outputs.items():
            node.listen(topic, msg_type)
        pubs = feed(node, None)
        procs = [start(c) for c in cmds]
        deadline = time.time() + timeout
        stamp = None
        while time.time() < deadline and any(v is None for v in node.received.values()):
            stamp = node.get_clock().now().to_msg()
            feed(node, (pubs, stamp))
            rclpy.spin_once(node, timeout_sec=0.1)
            if any(p.poll() is not None for p in procs):
                break
        extra = check(node) if all(v is not None for v in node.received.values()) else None
        results = [stop(p) for p in procs]
        logs = '\n'.join(out for _, out in results)
        died = [c for c, (alive, _) in zip(cmds, results) if not alive]
        missing = [t for t, v in node.received.items() if v is None]
        node.destroy_node()
        rclpy.shutdown()

        problems = []
        if died:
            problems.append('exited early: %s' % ', '.join(' '.join(c) for c in died))
        if missing:
            problems.append('no message on %s after %.0f s' % (', '.join(missing), timeout))
        if extra:
            problems.append(extra)
        if 'exception' in logs.lower() or '[ERROR]' in logs:
            problems.append('errors in the node log')
        emit(name, not problems, '\n'.join(problems) + ('\n' + logs if problems else ''))

    # -- opencv_demo and transport_demo: same contract, different plumbing ----------

    def feed_color(node, state):
        if state is None:
            return {'img': node.publisher('/color/image', Image)}
        pubs, stamp = state
        pubs['img'].publish(image(stamp, 'bgr8'))

    def check_gray(topic):
        def check(node):
            msg = node.received[topic]
            if (msg.width, msg.height, msg.encoding) != (W, H, 'mono8'):
                return 'expected %dx%d mono8, got %dx%d %s' % (W, H, msg.width, msg.height,
                                                               msg.encoding)
            if msg.header.frame_id != 'camera':
                return 'header not preserved: frame_id %r' % msg.header.frame_id
        return check

    scenario('opencv_demo: bgr8 in, mono8 out, header kept',
             [['ros2', 'run', 'opencv_demo', 'opencv_processing']],
             feed_color, {'/image_processed': Image}, check_gray('/image_processed'))

    scenario('transport_demo: same through image_transport',
             [['ros2', 'run', 'transport_demo', 'transport_processing']],
             feed_color, {'/image_processed': Image}, check_gray('/image_processed'))

    # -- sync_demo: no output topic, it shows the pair --------------------------------

    def feed_stereo(node, state):
        if state is None:
            return {'l': node.publisher('/left/image', Image),
                    'r': node.publisher('/right/image', Image)}
        pubs, stamp = state
        pubs['l'].publish(image(stamp, 'bgr8', 'left'))
        pubs['r'].publish(image(stamp, 'bgr8', 'right'))

    rclpy.init()
    node = Harness()
    pubs = feed_stereo(node, None)
    proc = start(['ros2', 'run', 'sync_demo', 'sync_processing'])
    deadline = time.time() + 30
    shown = False
    while time.time() < deadline and proc.poll() is None:
        feed_stereo(node, (pubs, node.get_clock().now().to_msg()))
        rclpy.spin_once(node, timeout_sec=0.1)
        tree = windows()
        if 'Left Image' in tree and 'Right Image' in tree:
            shown = True
            break
    alive, logs = stop(proc)
    node.destroy_node()
    rclpy.shutdown()
    problems = []
    if not alive:
        problems.append('exited early')
    if not shown:
        problems.append('the synchronized callback never opened its two windows')
    if 'exception' in logs.lower() or '[ERROR]' in logs:
        problems.append('errors in the node log')
    emit('sync_demo: paired callback runs', not problems,
         '\n'.join(problems) + ('\n' + logs if problems else ''))

    # -- pcl_demo -------------------------------------------------------------------

    def feed_cloud(node, state):
        if state is None:
            return {'pc': node.publisher('/stereo/points', PointCloud2)}
        pubs, stamp = state
        pubs['pc'].publish(cloud(stamp))

    def check_cloud(node):
        msg = node.received['/pcl_processed']
        if msg.width * msg.height != 100:
            return 'expected 100 points, got %d' % (msg.width * msg.height)
        if msg.header.frame_id != 'camera':
            return 'header not preserved: frame_id %r' % msg.header.frame_id

    scenario('pcl_demo: cloud in, same points out, header kept',
             [['ros2', 'run', 'pcl_demo', 'pcl_processing']],
             feed_cloud, {'/pcl_processed': PointCloud2}, check_cloud)

    # -- launch_demo + pcl_demo: the chain of the chapter ------------------------------

    def feed_depth(node, state):
        if state is None:
            return {'depth': node.publisher('/stereo/depth', Image),
                    'depth_info': node.publisher('/stereo/camera_info', CameraInfo),
                    'rgb': node.publisher('/color/image', Image),
                    'rgb_info': node.publisher('/color/camera_info', CameraInfo)}
        pubs, stamp = state
        pubs['depth'].publish(image(stamp, '16UC1'))
        pubs['depth_info'].publish(camera_info(stamp))
        pubs['rgb'].publish(image(stamp, 'rgb8'))
        pubs['rgb_info'].publish(camera_info(stamp))

    def check_chain(node):
        pts = node.received['/stereo/points']
        if pts.width * pts.height != W * H:
            return 'expected %d points from a %dx%d depth image, got %d' % (
                W * H, W, H, pts.width * pts.height)
        names = [f.name for f in pts.fields]
        if 'rgb' not in names:
            return 'cloud without colour: fields %s' % names

    scenario('launch_demo: depth to cloud, consumed by pcl_demo',
             [['ros2', 'launch', 'launch_demo', 'depth_processing.launch.py'],
              ['ros2', 'run', 'pcl_demo', 'pcl_processing']],
             feed_depth, {'/stereo/points': PointCloud2, '/pcl_processed': PointCloud2},
             check_chain, timeout=45.0)


# --- Main -------------------------------------------------------------------------

def main():
    if '--smoke-child' in sys.argv:
        smoke_child()
        return 0

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), help='repository to test (read only)')
    parser.add_argument('--work', default='/work', help='where the clean copy is made')
    parser.add_argument('--steps', default=','.join(STEPS),
                        help='comma-separated subset of: %s' % ', '.join(STEPS))
    parser.add_argument('--domain', type=int, default=42, help='ROS_DOMAIN_ID of the smoke test')
    parser.add_argument('--title', default='ROS 2', help='title of the summary')
    args = parser.parse_args()
    steps = args.steps.split(',')

    prepare(args.src, args.work)
    report = Report(STEPS)
    for step in STEPS:
        if step in steps:
            print('\n=== %s ===' % step, flush=True)
            globals()['step_' + step](args.work, report, args)

    summary = report.markdown(args.title)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(summary)
    failed = report.failed()
    print('\n%d checks, %d failed' % (len(report.rows), len(failed)))
    for step, name, _, _ in failed:
        print('  FAIL %s: %s' % (step, name))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
