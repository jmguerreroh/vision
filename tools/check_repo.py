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

# El capitulo 18 son paquetes de ROS 2: los construye colcon por nombre de
# paquete, no por carpeta numerada, asi que no se le aplican estas reglas.
CAP_ROS2 = '19_vision_ros2'


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

    # 6b. Todo nombre de fichero citado en el codigo existe bajo data/.
    # Va aparte de la comprobacion anterior porque hay ejemplos que arman la
    # ruta concatenando: 04_03 junta el directorio que recibe por argumento con
    # "Histogram_Comparison_Source_0.jpg", de modo que ninguna cadena del
    # fuente contiene la ruta entera. Buscar el nombre suelto es lo unico que
    # detecta que el fichero ya no esta
    disponibles = set()
    for base, _, ficheros in os.walk(os.path.join(RAIZ, 'data')):
        for f in ficheros:
            disponibles.add(f)
    for cap, ej in lista:
        for src in glob.glob(os.path.join(RAIZ, cap, ej, '*.cpp')):
            s = leer(src)
            for nombre in set(re.findall(
                    r'"([A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:jpg|jpeg|png|avi|mp4|ply|pcd))"', s)):
                if nombre not in disponibles and nombre not in s.split('imwrite')[0][:0]:
                    # solo interesa si el ejemplo lo LEE, no si lo escribe
                    if re.search(r'(imread|VideoCapture|loadPCDFile|readPLY|FileStorage)\b[^;]*'
                                 + re.escape(nombre), s) or ('/' not in nombre and
                                 re.search(r'\+\s*"' + re.escape(nombre) + r'"', s)):
                        fallos.append('%s: cita %s, que no esta bajo data/' % (ej, nombre))

    # 7. Cabecera de documentacion en todo el codigo, capitulo 18 incluido
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

    # Los listados del libro invocan los binarios por su nombre completo. Ese
    # nombre lleva dentro el numero del capitulo, asi que una renumeracion lo
    # deja obsoleto, y como vive dentro de un lstlisting no lo alcanza ninguna
    # comprobacion del lado del libro. Ya paso: dos ordenes del capitulo de
    # vision 3D siguieron invocando 10_03 y 11_03 despues de que el libro
    # pasara de 14 a 18 capitulos, y quien las copiaba obtenia un error.
    libro = os.path.join(os.path.dirname(RAIZ), 'cv_book', 'chapters')
    if os.path.isdir(libro):
        existentes = nombres          # los nombres de ejemplo, ya reunidos arriba
        for ruta in sorted(glob.glob(os.path.join(libro, 'chapter*.tex'))):
            texto = leer(ruta)
            for n_lin, linea in enumerate(texto.split('\n'), 1):
                for m in re.finditer(r'\./(\d{2}_\d{2}_[a-z0-9_]+)', linea):
                    if m.group(1) not in existentes:
                        fallos.append('%s:%d invoca ./%s, que no existe en el repositorio'
                                      % (os.path.relpath(ruta, os.path.dirname(RAIZ)),
                                         n_lin, m.group(1)))

        # Las rutas de datos que aparecen en la PROSA del libro no las alcanzaba
        # ninguna comprobacion: las de los listados y las del printHelp si, pero
        # una frase como "las monedas de data/coins.jpg" no. Asi sobrevivio a
        # cuatro auditorias que el fichero se llama coins.png, mientras el propio
        # ejemplo abria el .png correcto. Basta con exigir que todo data/algo.ext
        # escrito en \texttt{} exista de verdad.
        datos = os.path.join(RAIZ, 'data')
        for ruta in sorted(glob.glob(os.path.join(libro, '*.tex'))):
            texto = leer(ruta)
            for n_lin, linea in enumerate(texto.split('\n'), 1):
                for m in re.finditer(r'\\texttt\{[^}]*?data/([A-Za-z0-9_\\.-]+\.[A-Za-z0-9]{2,5})\}',
                                     linea):
                    nombre = m.group(1).replace('\\_', '_')
                    if not glob.glob(os.path.join(datos, '**', nombre),
                                     recursive=True):
                        fallos.append('%s:%d cita data/%s en la prosa, y no existe'
                                      % (os.path.relpath(ruta, os.path.dirname(RAIZ)),
                                         n_lin, nombre))

    # La cadena que imprime printHelp anuncia el fichero por defecto, pero el
    # que se usa de verdad es el del CommandLineParser. Las dos se escriben a
    # mano y en sitios distintos del fichero, asi que se separan sin que nada
    # falle: tres ejemplos anunciaban starry_night.jpg y abrian starry_night.png.
    # Como los dos ficheros existen, ni el programa ni la comprobacion de rutas
    # se enteraban; lo unico que quedaba mal era el --help.
    for f in sorted(glob.glob(os.path.join(RAIZ, '*', '*', 'main.cpp'))):
        rel = os.path.relpath(f, RAIZ)
        s_cpp = leer(f)
        # solo nombres de fichero: el radical lleva alguna letra y la extension
        # es alfabetica, para no confundir un 0.015 con un fichero
        FICHERO = r'([\w-]*[A-Za-z][\w-]*\.[A-Za-z]{2,4})'
        por_defecto = set(re.findall(r'\{@\w+\s*\|\s*\S*?' + FICHERO + r'\s*\|', s_cpp))
        anunciados = set(re.findall(r'default:\s*' + FICHERO + r'\s*\)', s_cpp))
        for nombre in sorted(anunciados - por_defecto):
            fallos.append('%s: la ayuda anuncia %s y el parser usa %s'
                          % (rel, nombre, ', '.join(sorted(por_defecto)) or 'otro'))

    # Los nucleos que el repositorio escribe a mano tienen que coincidir con los
    # que imprime el libro. Sobel es el caso que lo justifica: 03_02 definia sus
    # dos mascaras con el signo cambiado respecto del libro y de cv::Sobel, de
    # modo que el ejemplo devolvia el gradiente negado. Nada fallaba al
    # compilar ni al ejecutar, y el lector veia un signo en el capitulo 3 y el
    # contrario en el 7.
    NUCLEOS = {
        '04_pixel_and_filtering/04_02_convolution/main.cpp': {
            'createSobelXKernel': [-1, 0, 1, -2, 0, 2, -1, 0, 1],
            'createSobelYKernel': [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        },
    }
    for rel, funciones in NUCLEOS.items():
        ruta = os.path.join(RAIZ, rel)
        if not os.path.isfile(ruta):
            fallos.append('%s: no existe y hay nucleos declarados sobre el' % rel)
            continue
        s_cpp = leer(ruta)
        for funcion, esperado in funciones.items():
            m = re.search(re.escape(funcion) + r'\(\)\s*\{(.*?)\n\}', s_cpp, re.S)
            if not m:
                fallos.append('%s: no se encuentra %s' % (rel, funcion))
                continue
            visto = [int(x) for x in re.findall(r'-?\d+', m.group(1))
                     if x not in ('3', '32')][:len(esperado)]
            if visto != esperado:
                fallos.append('%s: %s vale %s y el libro escribe %s'
                              % (rel, funcion, visto, esperado))

    # El libro fija W (columnas) y H (filas) para las dimensiones de una imagen,
    # y reserva M y N para otras cosas. Dos ejemplos de frecuencia usaban M y N,
    # y ademas con sentidos opuestos entre si: en 05_01 N era el alto y en 05_02
    # era el ancho. Un lector que compare la formula del libro con la del
    # ejemplo encuentra letras distintas para lo mismo.
    for f in sorted(glob.glob(os.path.join(RAIZ, '*', '*', '*.cpp'))):
        rel = os.path.relpath(f, RAIZ)
        for n_lin, linea in enumerate(leer(f).split('\n'), 1):
            if re.search(r'\bint\s+[MN]\s*[,)=]', linea):
                fallos.append('%s:%d declara M o N como dimension; el libro usa W y H'
                              % (rel, n_lin))

    if fallos:
        print('\nFALLO: %d incoherencia(s)\n' % len(fallos))
        for f in fallos:
            print('  ' + f)
        return 1
    if not quiet:
        print('%d ejemplos comprobados.' % len(lista))
        print('OK: nombres, cabeceras, citas, rutas de datos y ayuda son coherentes.')
        print('OK: los binarios que invocan los listados del libro existen.')
        print('OK: la ayuda coincide con el parser y los nucleos con los del libro.')
        print('OK: las dimensiones se escriben W y H, como en el libro.')
        print('OK: las rutas data/ citadas en la prosa del libro existen.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
