#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprueba la coherencia del repositorio de ejemplos.

Casi todo lo que encontro la auditoria era mecanizable, y de la clase que
vuelve sola: nombres de ejecutable que se desincronizan al renombrar una
carpeta, cabeceras que citan un binario que ya no existe, capitulos
renumerados que dejan comentarios apuntando al numero antiguo. Este guion
comprueba justo eso, para que no haga falta descubrirlo dos veces.

Lo que NO comprueba es si el codigo hace lo que dice: eso se verifica
compilando y ejecutando, no leyendo.

Uso:
    python3 tools/check_repo.py            # informe completo
    python3 tools/check_repo.py --quiet    # solo el veredicto

Devuelve 0 si no hay infracciones y 1 si las hay.
"""

import glob
import io
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# El capitulo 14 son paquetes de ROS 2: los construye colcon por nombre de
# paquete, no por carpeta numerada, asi que no se le aplican estas reglas.
CAP_ROS2 = '14_vision_ros2'


def leer(ruta):
    return io.open(ruta, encoding='utf-8').read()


def ejemplos():
    """Devuelve [(carpeta_capitulo, carpeta_ejemplo)] ordenado."""
    salida = []
    for cap in sorted(os.listdir(RAIZ)):
        if not re.match(r'^\d\d_', cap) or cap == CAP_ROS2:
            continue
        for ej in sorted(os.listdir(os.path.join(RAIZ, cap))):
            if re.match(r'^\d\d_\d\d_', ej):
                salida.append((cap, ej))
    return salida


def main():
    quiet = '--quiet' in sys.argv
    fallos = []
    lista = ejemplos()
    nombres = {ej for _, ej in lista}

    cmake_raiz = leer(os.path.join(RAIZ, 'CMakeLists.txt'))
    declarados = set(re.findall(r'add_(?:cv|pcl)_example\(\s*(\S+?)[\s)]', cmake_raiz))
    declarados = {d.split('/')[-1] for d in declarados if not d.startswith('<')}

    # 1. Cobertura: cada ejemplo en disco se construye, y nada sobra
    for cap, ej in lista:
        if ej not in declarados:
            fallos.append('%s/%s no aparece en el CMakeLists.txt raiz' % (cap, ej))
    for d in sorted(declarados - nombres):
        fallos.append('el CMakeLists.txt raiz declara %s, que no existe en disco' % d)

    # 2. El binario se llama como su carpeta, se compile como se compile
    for cap, ej in lista:
        base = os.path.join(RAIZ, cap, ej)
        mk = os.path.join(base, 'Makefile')
        if os.path.exists(mk):
            for var, esperado in re.findall(r'^(TARGET\d?)\s*=\s*(.+)$', leer(mk), re.M):
                esperado = esperado.strip()
                valido = ('$(notdir $(CURDIR))', '$(notdir $(CURDIR))_frequencies')
                if esperado not in valido:
                    fallos.append('%s/Makefile: %s = %s (deberia derivarse de la carpeta)'
                                  % (ej, var, esperado))
        cm = os.path.join(base, 'CMakeLists.txt')
        if os.path.exists(cm):
            for tgt in re.findall(r'add_executable\((\S+)', leer(cm)):
                if tgt != ej:
                    fallos.append('%s/CMakeLists.txt genera "%s" en vez de "%s"'
                                  % (ej, tgt, ej))

    # 3. Las cabeceras no citan ejecutables que no existen
    for cap, ej in lista:
        for src in glob.glob(os.path.join(RAIZ, cap, ej, '*.cpp')):
            for n, linea in enumerate(leer(src).split('\n'), 1):
                for cita in re.findall(r'(?:Usage|Example):\s+\./(\S+)', linea):
                    if cita not in nombres and cita not in ('%s_frequencies' % ej,):
                        fallos.append('%s/%s:%d cita ./%s, que no es ningun ejecutable'
                                      % (ej, os.path.basename(src), n, cita))

    # 4. Toda cita NN_MM a otro ejemplo tiene que existir
    for cap, ej in lista:
        for src in glob.glob(os.path.join(RAIZ, cap, ej, '*.cpp')):
            for n, linea in enumerate(leer(src).split('\n'), 1):
                for cita in re.findall(r'\b(\d\d_\d\d_[a-z0-9_]+)', linea):
                    if cita not in nombres:
                        fallos.append('%s/%s:%d cita %s, que no existe'
                                      % (ej, os.path.basename(src), n, cita))

    # 5. Cada ejemplo acepta --help, y lo hace con el mismo patron
    for cap, ej in lista:
        src = os.path.join(RAIZ, cap, ej, 'main.cpp')
        if not os.path.exists(src):
            continue
        s = leer(src)
        if 'pcl::console' in s:
            if '"--help"' not in s or '"-h"' not in s:
                fallos.append('%s: ejemplo PCL que no acepta -h y --help' % ej)
        elif 'cv::CommandLineParser' in s:
            if 'parser.has("help")' not in s:
                fallos.append('%s: no atiende --help' % ej)
        else:
            fallos.append('%s: no usa ninguno de los dos parseadores' % ej)

    # 6. Las rutas de datos por defecto apuntan a ficheros que existen
    for cap, ej in lista:
        for src in glob.glob(os.path.join(RAIZ, cap, ej, '*.cpp')):
            for ruta in set(re.findall(r'\.\./\.\./(data/[A-Za-z0-9_./?*-]+)', leer(src))):
                completa = os.path.join(RAIZ, ruta)
                # Un prefijo que el codigo completa en ejecucion (result_000.pcd)
                # no nombra ningun fichero que se pueda comprobar aqui
                if ruta.endswith('_'):
                    continue
                if any(c in ruta for c in '*?'):
                    if not glob.glob(completa):
                        fallos.append('%s: el patron %s no encuentra ningun fichero' % (ej, ruta))
                elif not os.path.exists(completa):
                    fallos.append('%s: %s no existe' % (ej, ruta))

    # 7. Cabecera de documentacion en todo el codigo, capitulo 14 incluido
    fuentes = [f for f in glob.glob(os.path.join(RAIZ, '*', '*', '*.cpp')) +
               glob.glob(os.path.join(RAIZ, '*', '*', 'src', '*.cpp')) +
               glob.glob(os.path.join(RAIZ, '*', '*', 'include', '*', '*.hpp'))
               if '/old/' not in f and '/build/' not in f and '/install/' not in f]
    for f in fuentes:
        cab = leer(f)[:400]
        if '@file' not in cab or '@brief' not in cab:
            fallos.append('%s: sin cabecera @file/@brief'
                          % os.path.relpath(f, RAIZ))

    # 8. Restos de plantilla y anchura de linea
    for f in fuentes + glob.glob(os.path.join(RAIZ, CAP_ROS2, '*', 'package.xml')):
        rel = os.path.relpath(f, RAIZ)
        s = leer(f)
        if 'TODO' in s:
            fallos.append('%s: queda un TODO sin resolver' % rel)
        for n, linea in enumerate(s.split('\n'), 1):
            if len(linea) > 100 and not rel.endswith('.xml'):
                fallos.append('%s:%d pasa de 100 caracteres (%d)' % (rel, n, len(linea)))

    if fallos:
        print('\nFALLO: %d incoherencia(s)\n' % len(fallos))
        for f in fallos:
            print('  ' + f)
        return 1
    if not quiet:
        print('%d ejemplos comprobados.' % len(lista))
        print('OK: nombres, cabeceras, citas, rutas de datos y ayuda son coherentes.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
