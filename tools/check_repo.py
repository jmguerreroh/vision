#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the consistency of the examples repository.

Almost everything the audit found could be checked mechanically, and was of
the kind that comes back on its own: executable names that drift when a folder
is renamed, headers citing a binary that no longer exists, renumbered chapters
leaving comments that point at the old number. This script checks exactly
that, so nobody has to find it out twice.

What it does NOT check is whether the code does what it says: that is
verified by compiling and running, not by reading (see tools/ci/). Nor does it
read the sources of the book, which are not part of this repository: the
citations of the book are checked on the book side.

Usage:
    python3 tools/check_repo.py            # full report
    python3 tools/check_repo.py --quiet    # verdict only

Returns 0 if there are no violations and 1 if there are.
"""

import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chapter 19 is made of ROS 2 packages: colcon builds them by package name, not
# by numbered folder, so these rules do not apply to it.
ROS2_CHAPTER = '19_vision_ros2'


def read(path):
    return io.open(path, encoding='utf-8').read()


def examples():
    """Returns [(chapter_folder, example_folder)], sorted."""
    out = []
    for chap in sorted(os.listdir(ROOT)):
        if not re.match(r'^\d\d_', chap) or chap == ROS2_CHAPTER:
            continue
        for ex in sorted(os.listdir(os.path.join(ROOT, chap))):
            if re.match(r'^\d\d_\d\d_', ex):
                out.append((chap, ex))
    return out


def main():
    quiet = '--quiet' in sys.argv
    failures = []
    ex_list = examples()
    names = {ex for _, ex in ex_list}

    root_cmake = read(os.path.join(ROOT, 'CMakeLists.txt'))
    declared = set(re.findall(r'add_(?:cv|pcl)_example\(\s*(\S+?)[\s)]', root_cmake))
    declared = {d.split('/')[-1] for d in declared if not d.startswith('<')}

    # 1. Coverage: every example on disk is built, and nothing is left over
    for chap, ex in ex_list:
        if ex not in declared:
            failures.append('%s/%s is missing from the top-level CMakeLists.txt' % (chap, ex))
    for d in sorted(declared - names):
        failures.append('the top-level CMakeLists.txt declares %s, which does not exist on disk' % d)

    # 2. The binary is named after its folder, however it is compiled
    for chap, ex in ex_list:
        base = os.path.join(ROOT, chap, ex)
        mk = os.path.join(base, 'Makefile')
        if os.path.exists(mk):
            for var, expected in re.findall(r'^(TARGET\d?)\s*=\s*(.+)$', read(mk), re.M):
                expected = expected.strip()
                valid = ('$(notdir $(CURDIR))', '$(notdir $(CURDIR))_frequencies')
                if expected not in valid:
                    failures.append('%s/Makefile: %s = %s (should be derived from the folder)'
                                    % (ex, var, expected))
        cm = os.path.join(base, 'CMakeLists.txt')
        if os.path.exists(cm):
            for tgt in re.findall(r'add_executable\((\S+)', read(cm)):
                if tgt != ex:
                    failures.append('%s/CMakeLists.txt builds "%s" instead of "%s"'
                                    % (ex, tgt, ex))

    # 3. Headers do not cite executables that do not exist
    for chap, ex in ex_list:
        for src in glob.glob(os.path.join(ROOT, chap, ex, '*.cpp')):
            for n, line in enumerate(read(src).split('\n'), 1):
                for cited in re.findall(r'(?:Usage|Example):\s+\./(\S+)', line):
                    if cited not in names and cited not in ('%s_frequencies' % ex,):
                        failures.append('%s/%s:%d cites ./%s, which is not an executable'
                                        % (ex, os.path.basename(src), n, cited))

    # 3b. No file tells the reader to run a binary that does not exist.
    # Check 3 only looked at the line carrying "Usage:" or "Example:", and only
    # in the .cpp files. It missed six invocations of old names (./yolov4,
    # ./yolo11, ./semantic_segmentation) that lived on the following lines of
    # the same header and in the export_model.py scripts. This one looks at any
    # ./something, in .cpp, .py and .sh.
    ALLOWED_EXEC = {'download_model.sh', 'export_model.py', 'build'}
    invocation = re.compile(r'(?<![\w./])\./([A-Za-z0-9_][A-Za-z0-9_.-]*)')
    for chap, ex in ex_list:
        for pattern in ('*.cpp', '*.py', '*.sh'):
            for src in sorted(glob.glob(os.path.join(ROOT, chap, ex, pattern))):
                for n, line in enumerate(read(src).split('\n'), 1):
                    for cited in invocation.findall(line):
                        if cited in names or cited in ALLOWED_EXEC:
                            continue
                        if cited == '%s_frequencies' % ex:
                            continue
                        failures.append('%s/%s:%d tells to run ./%s, which is not an executable'
                                        % (ex, os.path.basename(src), n, cited))

    # 4. Every NN_MM citation of another example must exist
    for chap, ex in ex_list:
        for src in glob.glob(os.path.join(ROOT, chap, ex, '*.cpp')):
            for n, line in enumerate(read(src).split('\n'), 1):
                for cited in re.findall(r'\b(\d\d_\d\d_[a-z0-9_]+)', line):
                    if cited not in names:
                        failures.append('%s/%s:%d cites %s, which does not exist'
                                        % (ex, os.path.basename(src), n, cited))

    # 4b. Every bare NN_MM citation points at the example the comment describes.
    # Check 4 does not reach these: it requires the full name (NN_MM_something),
    # and a bare citation does not carry it. As a result the renumbering left 57
    # of 61 citations pointing at the neighbouring example without anything
    # failing, because the shifted number ALSO exists: the reader landed on a
    # real example, but about something else, and neither the compiler nor this
    # script said a word.
    #
    # The idea is to ask the citation to describe itself. If the comment names
    # an API function or a word that names some example, that evidence has to
    # be found in the cited example. "threshold in 09_01" fails because 09_01
    # is hough_lines; "threshold in 10_01" passes. Evidence coming from the
    # citing example itself is discarded: it describes the writer, not the
    # example being cited.
    #
    # What this check can NOT see: whether the wrongly cited example also uses
    # that function. "threshold in 09_01" passes, because 09_01_hough_lines
    # also binarizes before looking for lines. There is no mechanical way to
    # tell that case apart from "Sobel ... 04_02", which is right even though
    # the name of the example does not say Sobel.
    GENERIC = {'image', 'images', 'simple', 'advanced', 'read', 'write',
               'comparison', 'operations', 'transforms', 'code'}
    MODULES = {'ximgproc', 'aruco', 'surface_matching', 'viz', 'tracking'}

    # Citations the check flags that are correct, with the reason.
    ALLOWED = {
        # "Sure BACKGROUND" is watershed vocabulary, not a reference to
        # 16_04_background_subtraction. The word matches by chance.
        ('11_morphological_operations/11_08_distance_watershed/main.cpp', 20),
    }

    ex_source, ex_tokens = {}, {}
    for chap, ex in ex_list:
        text = ''
        for src in sorted(glob.glob(os.path.join(ROOT, chap, ex, '*.cpp'))):
            text += read(src)
        ex_source[ex[:5]] = text.lower()
        ex_tokens[ex[:5]] = {w for w in ex[6:].split('_')
                             if len(w) >= 3 and w not in GENERIC}
    all_tokens = set()
    for tk in ex_tokens.values():
        all_tokens |= tk

    def evidence(line):
        """What the comment claims about the example it cites."""
        # Without this, "\\nNote:" inside a literal reads as the word "nnote",
        # which belongs to nobody and flags the line for nothing.
        line = re.sub(r'\\[nrt]', ' ', line)
        ev = set(re.findall(r'cv::(\w+)', line))
        ev |= set(re.findall(r'\b([a-z]+[A-Z]\w*)\b', line))     # filter2D, inRange
        ev |= {w for w in re.findall(r'\b(\w+)\b', line) if w in MODULES}
        ev |= {w for w in re.findall(r'[a-zA-Z]{3,}', line.lower())
               if w in all_tokens}
        return {w.lower() for w in ev if len(w) >= 3 and w.lower() not in GENERIC}

    def supports(word, key):
        if word in ex_source[key]:
            return True
        return any(word.startswith(t) or t.startswith(word)
                   for t in ex_tokens[key])

    # A word found in half the examples tells nothing apart, and in fact it hid
    # a real bug: "no opencv_contrib needed (14_02 does need
    # ximgproc...)" passed because "opencv" appears in 68 of the 77 examples, so
    # it supported any of them. Only evidence that discriminates counts.
    def discriminates(word):
        return sum(1 for k in ex_source if supports(word, k)) <= len(ex_source) // 2

    reviewed = [os.path.join(ROOT, chap, ex, os.path.basename(s))
                for chap, ex in ex_list
                for s in sorted(glob.glob(os.path.join(ROOT, chap, ex, '*.cpp')))]
    reviewed += [os.path.join(ROOT, 'README.md'),
                 os.path.join(ROOT, 'tools', 'check_repo.py')]

    bare = re.compile(r'(?<![\w])(\d\d_\d\d)(?![\w\d])')
    for f in reviewed:
        rel = os.path.relpath(f, ROOT)
        own = None
        m_own = re.match(r'^\d\d_[a-z0-9_]+/(\d\d_\d\d)_', rel)
        if m_own:
            own = m_own.group(1)
        for n, line in enumerate(read(f).split('\n'), 1):
            for m in bare.finditer(line):
                num = m.group(1)
                if (rel, n) in ALLOWED:
                    continue
                if num not in ex_source:
                    failures.append('%s:%d cites %s, which is not an example'
                                    % (rel, n, num))
                    continue
                ev = evidence(line)
                if own:
                    ev -= ex_tokens[own] | {own}
                ev = {w for w in ev if discriminates(w)}
                if not ev or any(supports(w, num) for w in ev):
                    continue
                candidates = sorted(k for k in ex_source
                                    if all(supports(w, k) for w in ev))
                failures.append('%s:%d cites %s, but talks about %s (that would be %s)'
                                % (rel, n, num, ', '.join(sorted(ev)),
                                   ' or '.join(candidates[:3]) or 'no example'))

    # 4c. Every reference to a book chapter points at the chapter that deals
    # with that. It is the same defect as 4b, and it hid right next to it: the
    # renumbering shifted the chapters and the NN_MM citations of other
    # examples were fixed, but not the loose chapter numbers, which are a
    # different thing. 22 of 33 pointed at the neighbouring chapter. Two gave
    # themselves away, because the example number was right and the chapter
    # number was not: "are chapter 11 (12_01_region_moments)" and
    # "Chapter 10 (11_06_flood_fill)".
    #
    # Chapter N of the book is folder NN, so the vocabulary of a chapter is that
    # of all its examples together. The same criterion as in 4b applies:
    # evidence from the writing chapter itself does not count, and neither does
    # evidence found in nearly every chapter, because it tells nothing apart.
    # Words that name some folder and still do not identify a chapter:
    # "opencv" is in 15_04_opencv_icp and in the ROS 2 opencv_demo, "model" in
    # 15_10_pcl_ransac_model_fitting and "video" in 03_05_video_capture, but all
    # three show up when talking about anything.
    CHAP_GENERIC = {'opencv', 'model', 'video'}

    # Lines the check flags that are correct, with the reason.
    CHAP_ALLOWED = {
        # The sentence makes two claims: that THIS example belongs to chapter 8,
        # and that moments belong to chapter 12. The evidence of the second one
        # falls on the citation of the first.
        ('08_edge_detection/08_05_chain_code/main.cpp', 282),
    }

    chap_source, chap_tokens = {}, {}
    for chap, ex in ex_list:
        c = chap[:2]
        chap_source.setdefault(c, '')
        chap_tokens.setdefault(c, set())
        for src in sorted(glob.glob(os.path.join(ROOT, chap, ex, '*.cpp'))):
            chap_source[c] += read(src).lower()
        chap_tokens[c] |= ex_tokens[ex[:5]]
    # Chapter 19 is made of ROS 2 packages and does not go through examples():
    # it is assembled separately.
    ros2 = os.path.join(ROOT, ROS2_CHAPTER)
    if os.path.isdir(ros2):
        chap_source['19'] = ''
        chap_tokens['19'] = {'ros2', 'ros', 'launch', 'transport', 'sync', 'pcl', 'opencv'}
        for src in glob.glob(os.path.join(ros2, '*', 'src', '*.cpp')):
            chap_source['19'] += read(src).lower()

    # A chapter has up to thirteen examples, so their joint source contains
    # almost any word and tells nothing apart. What does tell chapters apart is
    # the folder names: "morphological" is only in chapter 11. Hence the two
    # levels: if the word names some example, the cited chapter must be one of
    # those carrying it in the name; if not (an API function, for instance),
    # finding it in the source is enough.
    def is_topic(word):
        for tk in chap_tokens.values():
            if any(word.startswith(x) or x.startswith(word) for x in tk):
                return True
        return False

    def supports_chap(word, c):
        if c not in chap_source:
            return False
        if any(word.startswith(x) or x.startswith(word) for x in chap_tokens[c]):
            return True
        if is_topic(word):
            return False          # a title word: the folder name decides
        return word in chap_source[c]

    def discriminates_chap(word):
        return sum(1 for c in chap_source if supports_chap(word, c)) <= len(chap_source) // 2

    reviewed_chap = []
    for chap, ex in ex_list:
        for pattern in ('*.cpp', '*.py', '*.sh'):
            reviewed_chap += sorted(glob.glob(os.path.join(ROOT, chap, ex, pattern)))
    for pattern in (('*', 'src', '*.cpp'), ('*', 'include', '*', '*.hpp'), ('*', 'launch', '*.py')):
        reviewed_chap += sorted(glob.glob(os.path.join(ros2, *pattern)))
    reviewed_chap += [os.path.join(ROOT, 'README.md'), os.path.join(ROOT, 'data', 'README.md'),
                      os.path.join(ROOT, 'CMakeLists.txt')]

    # "capítulo N" too: data/README.md is written in Spanish.
    chap_ref = re.compile(r'\b[Cc]hapter\s+(\d+)|\bcap[i\u00ed]tulo\s+(\d+)')
    for f in reviewed_chap:
        if not os.path.exists(f):
            continue
        rel = os.path.relpath(f, ROOT)
        own = rel[:2] if re.match(r'^\d\d_', rel) else None
        for n, line in enumerate(read(f).split('\n'), 1):
            for m in chap_ref.finditer(line):
                num = (m.group(1) or m.group(2)).zfill(2)
                if (rel, n) in CHAP_ALLOWED:
                    continue
                if num not in chap_source and num != '01':
                    failures.append('%s:%d cites chapter %s, which does not exist' % (rel, n, num))
                    continue
                ev = evidence(line)
                if own:
                    ev -= chap_tokens.get(own, set())
                ev = {w for w in ev - CHAP_GENERIC if discriminates_chap(w)}
                if not ev or any(supports_chap(w, num) for w in ev):
                    continue
                candidates = sorted(c for c in chap_source
                                    if all(supports_chap(w, c) for w in ev))
                failures.append('%s:%d cites chapter %s, but talks about %s (that would be %s)'
                                % (rel, n, num, ', '.join(sorted(ev)),
                                   ' or '.join(candidates[:3]) or 'none'))

    # 4d. The ROS 2 package.xml files cite their own chapter. Their
    # <description> is what `ros2 pkg xml` shows, and all five kept saying
    # "Chapter 18" after ROS 2 moved to 19, without anything noticing. This is
    # not done with the heuristic of 4c, which guesses the topic from words and
    # gets it wrong here: it takes "chain", "Mat" or "PointCloud2" for topics of
    # chapters 8, 3 or 15. The exact rule is simpler: all these packages belong
    # to a single chapter.
    ros2_own = str(int(ROS2_CHAPTER[:2]))
    for f in sorted(glob.glob(os.path.join(ROOT, ROS2_CHAPTER, '*', 'package.xml'))):
        rel = os.path.relpath(f, ROOT)
        for n, line in enumerate(read(f).split('\n'), 1):
            for m in chap_ref.finditer(line):
                num = m.group(1) or m.group(2)
                if num != ros2_own:
                    failures.append('%s:%d cites chapter %s; the packages of %s belong to %s'
                                    % (rel, n, num, ROS2_CHAPTER, ros2_own))

    # 5. Every example accepts --help, and with the same pattern
    for chap, ex in ex_list:
        src = os.path.join(ROOT, chap, ex, 'main.cpp')
        if not os.path.exists(src):
            continue
        s = read(src)
        if 'pcl::console' in s:
            if '"--help"' not in s or '"-h"' not in s:
                failures.append('%s: PCL example that does not accept -h and --help' % ex)
        elif 'cv::CommandLineParser' in s:
            if 'parser.has("help")' not in s:
                failures.append('%s: does not handle --help' % ex)
        else:
            failures.append('%s: uses neither of the two parsers' % ex)

    # 6. The default data paths point at files that exist
    for chap, ex in ex_list:
        for src in glob.glob(os.path.join(ROOT, chap, ex, '*.cpp')):
            for path in set(re.findall(r'\.\./\.\./(data/[A-Za-z0-9_./?*-]+)', read(src))):
                full = os.path.join(ROOT, path)
                # A prefix the code completes at run time (result_000.pcd) does
                # not name any file that can be checked here
                if path.endswith('_'):
                    continue
                if any(c in path for c in '*?'):
                    if not glob.glob(full):
                        failures.append('%s: the pattern %s matches no file' % (ex, path))
                elif not os.path.exists(full):
                    failures.append('%s: %s does not exist' % (ex, path))

    # 6b. Every file name cited in the code exists under data/.
    # Separate from the previous check because some examples build the path by
    # concatenation: 05_03 joins the directory it gets as an argument with
    # "Histogram_Comparison_Source_0.jpg", so no string in the source holds the
    # whole path. Looking for the bare name is the only way to detect that the
    # file is gone
    available = set()
    for base, _, files in os.walk(os.path.join(ROOT, 'data')):
        for f in files:
            available.add(f)
    for chap, ex in ex_list:
        for src in glob.glob(os.path.join(ROOT, chap, ex, '*.cpp')):
            s = read(src)
            for name in set(re.findall(
                    r'"([A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:jpg|jpeg|png|avi|mp4|ply|pcd))"', s)):
                if name not in available and name not in s.split('imwrite')[0][:0]:
                    # it only matters if the example READS it, not if it writes it
                    if re.search(r'(imread|VideoCapture|loadPCDFile|readPLY|FileStorage)\b[^;]*'
                                 + re.escape(name), s) or ('/' not in name and
                                 re.search(r'\+\s*"' + re.escape(name) + r'"', s)):
                        failures.append('%s: cites %s, which is not under data/' % (ex, name))

    # 7. Documentation header in all the code, chapter 19 included
    sources = [f for f in glob.glob(os.path.join(ROOT, '*', '*', '*.cpp')) +
               glob.glob(os.path.join(ROOT, '*', '*', 'src', '*.cpp')) +
               glob.glob(os.path.join(ROOT, '*', '*', 'include', '*', '*.hpp'))
               if '/old/' not in f and '/build/' not in f and '/install/' not in f]
    for f in sources:
        head = read(f)[:400]
        if '@file' not in head or '@brief' not in head:
            failures.append('%s: no @file/@brief header'
                            % os.path.relpath(f, ROOT))

    # 8. Template leftovers and line width
    for f in sources + glob.glob(os.path.join(ROOT, ROS2_CHAPTER, '*', 'package.xml')):
        rel = os.path.relpath(f, ROOT)
        s = read(f)
        if 'TODO' in s:
            failures.append('%s: an unresolved TODO is left' % rel)
        for n, line in enumerate(s.split('\n'), 1):
            if len(line) > 100 and not rel.endswith('.xml'):
                failures.append('%s:%d is longer than 100 characters (%d)' % (rel, n, len(line)))

    # The string printHelp prints announces the default file, but the one really
    # used is the CommandLineParser's. Both are written by hand, in different
    # places of the file, so they drift apart without anything failing: three
    # examples announced starry_night.jpg and opened starry_night.png. Since
    # both files exist, neither the program nor the path check noticed; the
    # only thing wrong was the --help.
    for f in sorted(glob.glob(os.path.join(ROOT, '*', '*', 'main.cpp'))):
        rel = os.path.relpath(f, ROOT)
        s_cpp = read(f)
        # file names only: the stem has some letter and the extension is
        # alphabetic, so that a 0.015 is not taken for a file
        FILE = r'([\w-]*[A-Za-z][\w-]*\.[A-Za-z]{2,4})'
        defaults = set(re.findall(r'\{@\w+\s*\|\s*\S*?' + FILE + r'\s*\|', s_cpp))
        announced = set(re.findall(r'default:\s*' + FILE + r'\s*\)', s_cpp))
        for name in sorted(announced - defaults):
            failures.append('%s: the help announces %s and the parser uses %s'
                            % (rel, name, ', '.join(sorted(defaults)) or 'another one'))

    # The kernels the repository writes by hand must match the ones the book
    # prints. Sobel is the case that justifies it: 04_02 defined its two masks
    # with the sign flipped with respect to the book and to cv::Sobel, so the
    # example returned the negated gradient. Nothing failed to compile or to
    # run, and the reader saw one sign in chapter 4 and the opposite in 8.
    KERNELS = {
        '04_pixel_and_filtering/04_02_convolution/main.cpp': {
            'createSobelXKernel': [-1, 0, 1, -2, 0, 2, -1, 0, 1],
            'createSobelYKernel': [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        },
    }
    for rel, functions in KERNELS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            failures.append('%s: does not exist and kernels are declared on it' % rel)
            continue
        s_cpp = read(path)
        for function, expected in functions.items():
            m = re.search(re.escape(function) + r'\(\)\s*\{(.*?)\n\}', s_cpp, re.S)
            if not m:
                failures.append('%s: %s not found' % (rel, function))
                continue
            seen = [int(x) for x in re.findall(r'-?\d+', m.group(1))
                    if x not in ('3', '32')][:len(expected)]
            if seen != expected:
                failures.append('%s: %s is %s and the book writes %s'
                                % (rel, function, seen, expected))

    # The book uses W (columns) and H (rows) for the dimensions of an image, and
    # keeps M and N for other things. Two frequency examples used M and N, and
    # with opposite meanings at that: in 06_01 N was the height and in 06_02 it
    # was the width. A reader comparing the formula of the book with the one of
    # the example finds different letters for the same thing.
    for f in sorted(glob.glob(os.path.join(ROOT, '*', '*', '*.cpp'))):
        rel = os.path.relpath(f, ROOT)
        for n_line, line in enumerate(read(f).split('\n'), 1):
            if re.search(r'\bint\s+[MN]\s*[,)=]', line):
                failures.append('%s:%d declares M or N as a dimension; the book uses W and H'
                                % (rel, n_line))

    if failures:
        print('\nFAILED: %d inconsistenc%s\n' % (len(failures),
                                                 'y' if len(failures) == 1 else 'ies'))
        for f in failures:
            print('  ' + f)
        return 1
    if not quiet:
        print('%d examples checked.' % len(ex_list))
        print('OK: names, headers, citations, data paths and help are consistent.')
        print('OK: every NN_MM citation points at the example the comment describes.')
        print('OK: every chapter reference points at the chapter that deals with it.')
        print('OK: the help matches the parser, and the kernels match the book.')
        print('OK: dimensions are written W and H, as in the book.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
