# Procedencia y licencias de los datos

Los programas de ejemplo leen sus imágenes y vídeos de esta carpeta. La mayor parte
procede de colecciones públicas estándar del área; el resto es material propio. El
repositorio se distribuye bajo MIT, pero **esa licencia cubre el código, no estos
datos**: cada conjunto conserva la licencia de su fuente original.

| Conjunto | Ficheros | Fuente | Licencia / condiciones |
|---|---|---|---|
| Muestras de OpenCV | `baboon.jpg`, `fruits.jpg`, `messi5.jpg`, `sudoku.png`, `board.jpg`, `blox.jpg`, `box.png`, `box_in_scene.png`, `digits.png`, `pic1..6.png`, `stuff.jpg`, `home.jpg`, `chicky_512.png`, `HappyFish.jpg`, `LinuxLogo.jpg`, `WindowsLogo.jpg`, `opencv-logo*.png`, `vtest.avi`, `tree.avi`, `Megamind*.avi`, `left*.jpg`, `right*.jpg`, `left_intrinsics.yml`, `intrinsics.yml`, `stereo_calib.xml`, `letter-recognition.data`, `parasaurolophus_*.ply`, `ml.png`, `templ.png`, `lena*.jpg` | [opencv/opencv_extra](https://github.com/opencv/opencv) (carpetas `samples/data`) | Apache 2.0 |
| Par estéreo Aloe | `aloeL.jpg`, `aloeR.jpg`, `aloeGT.png` | Middlebury Stereo 2006 (Scharstein y Pal, CVPR 2007) | uso académico con cita de la fuente |
| Flujo óptico | `rubberwhale1.png`, `rubberwhale2.png` | Middlebury Optical Flow (Baker et al.) | uso académico con cita de la fuente |
| Pares con homografía | `graf1.png`, `graf3.png`, `H1to3p.xml`, `leuvenA.jpg`, `leuvenB.jpg` | Oxford VGG (Mikolajczyk y Schmid) | uso académico con cita de la fuente |
| Renders de Blender | `Blender_Suzanne1.jpg`, `Blender_Suzanne2.jpg` | renders propios del modelo Suzanne de Blender | CC0 del modelo; renders propios |
| Fotografias del libro | `aerial_view.png`, `building_facade.png`, `chess.png`, `coins.png`, `futbol.png`, `maize.jpg`, `smarties.png`, `starry_night.png` | autor del libro | MIT, como el resto del repositorio |
| Material propio | `page_uneven.png`, `test_chart.png`, `coins.jpg`, `create_checkerboard.py`, `create_shapes.py`, `shapes.png`, `checkerboard.png`, `calibration_images/`, `aruco/`, `pcl_data/`, ficheros `.yml`/`.xml` generados por los ejemplos | autor del libro | MIT, como el resto del repositorio |
| Modelos DNN | `dnn/`, `models/` | descargados por los guiones de cada ejemplo (Darknet, Ultralytics) | licencia de cada modelo (consultar su fuente) |

Notas:

- Las fotografias de la fila «Fotografias del libro» son las mismas que usan los
  guiones de `~/cv_book/tools` para generar las figuras. Un ejemplo lanzado sin
  argumentos trabaja por tanto sobre la misma imagen que el lector acaba de ver
  impresa. Las versiones anteriores (`aero1.jpg`, `chess.jpg`,
  `starry_night.jpg`, `coins.jpg`, `building.jpg`) siguen aqui porque otros
  ejemplos las usan y porque sirven para comprobar que un ejemplo no depende de
  una resolucion concreta.

- `lena.jpg` y `lena_tmpl.jpg` no los usa ningún ejemplo y su uso está hoy desaconsejado
  en publicaciones; pueden eliminarse sin efecto sobre el repositorio.
- Si un fichero no aparece en la tabla, debe asumirse procedencia OpenCV salvo
  verificación en contra; conviene revisar esta tabla antes de cualquier redistribución
  comercial.
