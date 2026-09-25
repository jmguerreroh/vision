#!/usr/bin/env python3
"""
Checks, on a clean copy of the repository, what the README promises.

Steps, in order (select some with --steps):

  build       the top-level CMake build: every target, no compiler warning,
              no CMake warning from the repository's own CMake files, and the
              models of chapter 18 in place
  check       tools/check_repo.py, the static checks. After the build, because
              it requires the model files the build generates
  standalone  each example built on its own, with its Makefile or its
              CMakeLists.txt, into a binary named after its folder
  help        every binary answers -h and --help with exit code 0
  run         every example with its default input, under the autopilot
              (tools/ci/autopilot.cpp), from vision_examples/bin/ and from its
              own folder, the two places the README says it works from
  clean       what the build and the runs wrote is ignored by .gitignore, and
              no versioned file changed (15_05 must rewrite test_pcd.pcd byte
              for byte)

It runs inside the image of tools/ci/Dockerfile, which provides the
libraries, Xvfb and /opt/autopilot/libautopilot.so:

  xvfb-run -a python3 tools/ci/run_examples.py --src /src --work /work

--src is the repository to test, never written to. --work receives a copy of
its versioned and untracked-but-not-ignored files, which is what a fresh clone
plus the local changes would contain, and that copy is committed to a scratch
git repository so the clean step can diff against it.

Exit code 0 if everything passed, 1 otherwise.
"""

import argparse
import concurrent.futures
import glob
import os
import re
import shutil
import subprocess
import sys
import time

AUTOPILOT = '/opt/autopilot/libautopilot.so'
AUTOPILOT_PCL = '/opt/autopilot/libautopilot_pcl.so'
ROS2_CHAPTER = '19_vision_ros2'
STEPS = ['build', 'check', 'standalone', 'help', 'run', 'clean']

# Arguments for the runs. An example not listed runs once with no arguments.
# Each entry is a list of argument lists: one run per list.
RUN_ARGS = {
    # With no argument these two open the camera, which a runner does not have.
    '03_05_video_capture': [['../../data/vtest.avi']],
    '06_03_wavelet_denoising': [['../../data/starry_night.png']],
    # A menu of seven demos: all of them. With no option it prints the menu.
    '15_08_pcl_advanced_visualizer': [['-s'], ['-c'], ['-r'], ['-n'], ['-a'], ['-v'], ['-i']],
}

# Runs that need another example to have run first: they go in a second pass.
# 15_03 with the calibration 14_03 writes, on a pair of the same rig.
LINKED_RUNS = [
    ('15_03_stereo_to_pointcloud',
     ['../../data/left01.jpg', '../../data/right01.jpg', '--calib=stereo_calibration.yml'],
     '14_03_stereo_calibration'),
]

# Model files the build must leave in data/models/ (README, "Deep learning
# models"). Their absence fails the build step when --require-models is given.
MODELS = {
    '18_01_yolov4_darknet': ['yolov4/yolov4-tiny.weights', 'yolov4/yolov4-tiny.cfg'],
    '18_02_yolo_ultralytics': ['yolo11/yolo11n.onnx'],
    '18_03_semantic_segmentation': ['deeplabv3/deeplabv3_mobilenetv3.onnx'],
}


# --- Small helpers ----------------------------------------------------------------

class Report:
    """Collects PASS/FAIL/SKIP lines per step, and prints them as they come."""

    def __init__(self, steps):
        self.steps = steps
        self.rows = []

    def add(self, step, name, status, detail=''):
        self.rows.append((step, name, status, detail))
        mark = {'PASS': 'ok  ', 'FAIL': 'FAIL', 'SKIP': 'skip'}[status]
        line = '  [%s] %s' % (mark, name)
        if detail and status != 'PASS':
            line += '\n' + '\n'.join('         ' + d for d in detail.splitlines()[-15:])
        elif detail:
            line += '  (%s)' % detail
        print(line, flush=True)

    def failed(self):
        return [r for r in self.rows if r[2] == 'FAIL']

    def markdown(self, title):
        out = ['## %s' % title, '']
        for step in self.steps:
            rows = [r for r in self.rows if r[0] == step]
            if not rows:
                continue
            n_fail = sum(r[2] == 'FAIL' for r in rows)
            n_skip = sum(r[2] == 'SKIP' for r in rows)
            icon = ':x:' if n_fail else ':white_check_mark:'
            out.append('%s **%s**: %d passed, %d failed, %d skipped'
                       % (icon, step, len(rows) - n_fail - n_skip, n_fail, n_skip))
            for r in rows:
                if r[2] != 'PASS':
                    out.append('  - %s `%s`: %s' % (r[2], r[1], r[3].splitlines()[-1] if r[3] else ''))
        return '\n'.join(out) + '\n'


def run(cmd, cwd, timeout=None, env=None):
    """Runs cmd and returns (exit code, combined output, seconds). 124 = timeout."""
    start = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL)
        code, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        code, out = 124, (e.stdout or b'')
    return code, out.decode('utf-8', 'replace'), time.time() - start


def describe(code):
    if code == 124:
        return 'timed out'
    if code < 0:
        return 'killed by signal %d' % -code
    return 'exit code %d' % code


def examples(root):
    """[(chapter, example)] of the numbered examples, as check_repo.py sees them."""
    out = []
    for chap in sorted(os.listdir(root)):
        if not re.match(r'^\d\d_', chap) or chap == ROS2_CHAPTER:
            continue
        for ex in sorted(os.listdir(os.path.join(root, chap))):
            if re.match(r'^\d\d_\d\d_', ex):
                out.append((chap, ex))
    return out


def expected_binaries(root):
    """Binaries of the top-level build: main.cpp -> <dir>, other.cpp -> <dir>_<stem>."""
    out = {}
    for chap, ex in examples(root):
        for src in sorted(glob.glob(os.path.join(root, chap, ex, '*.cpp'))):
            stem = os.path.splitext(os.path.basename(src))[0]
            out[ex if stem == 'main' else '%s_%s' % (ex, stem)] = (chap, ex)
    return out


def own_cmake_warnings(out, root):
    """CMake warnings raised from the repository's own files. Those raised by the
    CMake modules of the system (PCL's FindFLANN, for one) are not its to fix."""
    warnings = []
    for m in re.finditer(r'^CMake Warning(?: \(dev\))?(?: at (\S+?):\d+)?.*$', out, re.M):
        where = m.group(1)
        if where is None or not os.path.isabs(where) or where.startswith(root):
            warnings.append(m.group(0))
    return warnings


def opencv_version():
    code, out, _ = run(['pkg-config', '--modversion', 'opencv4'], cwd='/')
    return tuple(int(x) for x in out.strip().split('.')[:2]) if code == 0 else (0, 0)


# --- Steps ------------------------------------------------------------------------

def prepare(src, work):
    """Copies the repository as a fresh clone plus local changes would have it."""
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    files = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=src).decode().split('\0')
    for f in filter(None, files):
        if not os.path.lexists(os.path.join(src, f)):
            continue  # deleted locally but still in the index
        os.makedirs(os.path.dirname(os.path.join(work, f)) or work, exist_ok=True)
        shutil.copy2(os.path.join(src, f), os.path.join(work, f), follow_symlinks=False)
    git = ['git', '-c', 'user.name=ci', '-c', 'user.email=ci@localhost']
    subprocess.check_call(git + ['init', '-q'], cwd=work)
    subprocess.check_call(git + ['add', '-A'], cwd=work)
    subprocess.check_call(git + ['commit', '-q', '-m', 'baseline'], cwd=work)


def step_check(root, report, args):
    code, out, _ = run([sys.executable, 'tools/check_repo.py'], cwd=root)
    report.add('check', 'tools/check_repo.py', 'PASS' if code == 0 else 'FAIL',
               '' if code == 0 else out)


def step_build(root, report, args):
    build = os.path.join(root, 'vision_examples', 'build')
    code, out, _ = run(['cmake', '-B', build], cwd=root)
    if code != 0:
        report.add('build', 'cmake configure', 'FAIL', out)
        return
    warnings = own_cmake_warnings(out, root)
    report.add('build', 'cmake configure', 'FAIL' if warnings else 'PASS',
               '\n'.join(warnings) + ('\n' + out if warnings else ''))

    code, out, secs = run(['cmake', '--build', build, '-j', str(args.jobs)], cwd=root)
    if code != 0:
        errors = [l for l in out.splitlines() if 'error' in l.lower()]
        report.add('build', 'cmake --build', 'FAIL', '\n'.join(errors[:30]) or out)
        return
    warnings = [l for l in out.splitlines() if re.search(r'\bwarning:', l)]
    report.add('build', 'cmake --build', 'FAIL' if warnings else 'PASS',
               '\n'.join(warnings) if warnings else '%.0f s' % secs)

    bin_dir = os.path.join(root, 'vision_examples', 'bin')
    missing = [b for b in expected_binaries(root) if not os.path.isfile(os.path.join(bin_dir, b))]
    extra = sorted(set(os.listdir(bin_dir)) - set(expected_binaries(root)))
    report.add('build', 'one binary per source in vision_examples/bin/',
               'FAIL' if missing or extra else 'PASS',
               'missing: %s\nunexpected: %s' % (missing, extra) if missing or extra
               else '%d binaries' % len(expected_binaries(root)))

    for ex, files in MODELS.items():
        absent = [f for f in files
                  if not os.path.isfile(os.path.join(root, 'data', 'models', f))]
        status = 'PASS' if not absent else ('FAIL' if args.require_models else 'SKIP')
        report.add('build', 'model of %s' % ex, status,
                   'not generated: %s' % ', '.join(absent) if absent else '')


def build_one(root, chap, ex):
    d = os.path.join(root, chap, ex)
    if os.path.isfile(os.path.join(d, 'Makefile')):
        code, out, _ = run(['make'], cwd=d)
        binary = os.path.join(d, ex)
    elif os.path.isfile(os.path.join(d, 'CMakeLists.txt')):
        code, out, _ = run(['cmake', '-B', 'build'], cwd=d)
        if code == 0:
            code, out2, _ = run(['cmake', '--build', 'build'], cwd=d)
            out += out2
        binary = os.path.join(d, 'build', ex)
    else:
        return 'FAIL', 'neither Makefile nor CMakeLists.txt'
    if code != 0:
        return 'FAIL', out
    if not os.path.isfile(binary):
        return 'FAIL', 'built, but %s does not exist' % os.path.relpath(binary, root)
    warnings = [l for l in out.splitlines() if re.search(r'\bwarning:', l)]
    return ('FAIL', '\n'.join(warnings)) if warnings else ('PASS', '')


def step_standalone(root, report, args):
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = {(chap, ex): pool.submit(build_one, root, chap, ex)
                   for chap, ex in examples(root)}
        for (chap, ex), fut in futures.items():
            status, detail = fut.result()
            report.add('standalone', '%s/%s' % (chap, ex), status, detail)


def step_help(root, report, args):
    bin_dir = os.path.join(root, 'vision_examples', 'bin')
    for binary in sorted(expected_binaries(root)):
        path = os.path.join(bin_dir, binary)
        if not os.path.isfile(path):
            report.add('help', binary, 'FAIL', 'not built')
            continue
        problems = []
        for flag in ('-h', '--help'):
            # No display: asking for help must not need one.
            env = dict(os.environ)
            env.pop('DISPLAY', None)
            code, out, _ = run([path, flag], cwd=bin_dir, timeout=20, env=env)
            if code != 0 or not out.strip():
                problems.append('%s: %s%s' % (flag, describe(code),
                                              '' if out.strip() else ', no output'))
        report.add('help', binary, 'FAIL' if problems else 'PASS', '\n'.join(problems))


def needs_pcl_autopilot(path):
    code, out, _ = run(['readelf', '-d', path], cwd='/')
    return 'libpcl_visualization' in out


def run_example(root, binary, argv, cwd, args):
    path = os.path.join(root, 'vision_examples', 'bin', binary)
    env = dict(os.environ,
               LD_PRELOAD=AUTOPILOT_PCL if needs_pcl_autopilot(path) else AUTOPILOT,
               OPENCV_SAMPLES_DATA_PATH=os.path.join(root, 'data') + '/')
    return run([path] + argv, cwd=cwd, timeout=args.timeout, env=env)


def verdict(binary, code, out, cv_version):
    """PASS, FAIL or SKIP for one run, knowing the exceptions the README lists."""
    if 'missing symbol' in out and '[autopilot]' in out:
        return 'FAIL', out
    if binary == '18_02_yolo_ultralytics' and cv_version < (4, 9):
        # README: YOLO11 needs OpenCV >= 4.9; with an older one the example
        # must say so and exit, not crash.
        if code == 1 and '4.9' in out:
            return 'PASS', 'refused OpenCV %d.%d as documented' % cv_version
        return 'FAIL', 'expected the OpenCV >= 4.9 message and exit 1, got %s\n%s' % (describe(code), out)
    if code == 0:
        return 'PASS', ''
    return 'FAIL', '%s\n%s' % (describe(code), out)


def step_run(root, report, args):
    cv_version = opencv_version()
    bin_dir = os.path.join(root, 'vision_examples', 'bin')
    binaries = expected_binaries(root)
    jobs = []
    for binary, (chap, ex) in sorted(binaries.items()):
        if args.only and not re.search(args.only, binary):
            continue
        for argv in RUN_ARGS.get(binary, [[]]):
            for where in args.cwd:
                cwd = bin_dir if where == 'bin' else os.path.join(root, chap, ex)
                jobs.append((binary, argv, where, cwd))

    def label(binary, argv, where):
        return ' '.join([binary] + argv) + ('   [from %s]' % ('bin/' if where == 'bin' else 'its folder'))

    with concurrent.futures.ThreadPoolExecutor(args.run_jobs) as pool:
        futures = [(j, pool.submit(run_example, root, j[0], j[1], j[3], args)) for j in jobs]
        for (binary, argv, where, cwd), fut in futures:
            code, out, secs = fut.result()
            status, detail = verdict(binary, code, out, cv_version)
            report.add('run', label(binary, argv, where), status,
                       detail if status != 'PASS' else (detail or '%.1f s' % secs))

    # Second pass: runs that consume what another one wrote.
    for binary, argv, producer in LINKED_RUNS:
        if args.only and not re.search(args.only, binary):
            continue
        if 'bin' not in args.cwd:
            continue
        if args.only and not re.search(args.only, producer):
            run_example(root, producer, [], bin_dir, args)
        code, out, secs = run_example(root, binary, argv, bin_dir, args)
        status, detail = verdict(binary, code, out, cv_version)
        report.add('run', label(binary, argv, 'bin') + ' after ' + producer, status,
                   detail if status != 'PASS' else '%.1f s' % secs)


def step_clean(root, report, args):
    code, out, _ = run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root)
    dirty = [l for l in out.splitlines() if l.strip()]
    modified = [l for l in dirty if not l.startswith('??')]
    untracked = [l for l in dirty if l.startswith('??')]
    report.add('clean', 'versioned files unchanged', 'FAIL' if modified else 'PASS',
               '\n'.join(modified))
    report.add('clean', 'everything written is in .gitignore', 'FAIL' if untracked else 'PASS',
               '\n'.join(untracked))


# --- Main -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), help='repository to test (read only)')
    parser.add_argument('--work', default='/work', help='where the clean copy is made')
    parser.add_argument('--in-place', action='store_true',
                        help='test --src itself instead of a copy (no clean step)')
    parser.add_argument('--steps', default=','.join(STEPS),
                        help='comma-separated subset of: %s' % ', '.join(STEPS))
    parser.add_argument('--only', help='regular expression: run only the matching binaries')
    parser.add_argument('--cwd', default='bin,example',
                        help='where to run the examples from: bin, example or both')
    parser.add_argument('--timeout', type=int, default=180, help='seconds per run')
    parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='parallel compilations')
    parser.add_argument('--run-jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help='parallel example runs')
    parser.add_argument('--require-models', action='store_true',
                        help='fail, instead of skip, when a chapter 18 model was not generated')
    parser.add_argument('--title', default='Examples', help='title of the summary')
    args = parser.parse_args()
    args.cwd = args.cwd.split(',')
    steps = args.steps.split(',')
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error('unknown steps: %s' % ', '.join(sorted(unknown)))

    if args.in_place:
        root = args.src
        steps = [s for s in steps if s != 'clean']
    else:
        root = args.work
        prepare(args.src, root)

    if 'run' in steps and not os.environ.get('DISPLAY'):
        parser.error('the run step needs a display: launch under xvfb-run -a')
    if 'run' in steps and not os.path.isfile(AUTOPILOT):
        parser.error('%s not found: run inside the image of tools/ci/Dockerfile' % AUTOPILOT)

    report = Report(STEPS)
    for step in STEPS:
        if step in steps:
            print('\n=== %s ===' % step, flush=True)
            globals()['step_' + step](root, report, args)

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
