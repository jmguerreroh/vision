# Procedencia y licencias de los datos

Los programas de ejemplo leen sus imágenes y vídeos de esta carpeta. **Aquí solo
está lo que usa algún ejemplo**: el material que dejó de usarse, casi todo muestras
de OpenCV, vivió en `old_data/` hasta el 23-09-2026 y se retiró del árbol publicado
porque ningún ejemplo lo lee; sigue en el historial de git. La mayor parte procede
de colecciones públicas estándar del área; el
resto es material propio. El repositorio se distribuye bajo MIT, pero **esa
licencia cubre el código, no estos datos**: cada conjunto conserva la licencia de
su fuente original.

| Conjunto | Ficheros | Fuente | Licencia / condiciones |
|---|---|---|---|
| Muestras de OpenCV | `vtest.avi`, `Megamind.avi`, `digits.png`, `form.jpg`, `scanned-form.jpg`, `fruits.jpg`, `star.jpg`, `RGB.jpg`, `left*.jpg`, `right*.jpg`, `left.jpg`, `right.jpg`, `parasaurolophus_*.ply` | [opencv/opencv_extra](https://github.com/opencv/opencv) (carpetas `samples/data`) | Apache 2.0 |
| Par estéreo Aloe | `aloeL.jpg`, `aloeR.jpg`, `aloeGT.png` | Middlebury Stereo 2006 (Scharstein y Pal, CVPR 2007) | uso académico con cita de la fuente |
| Fotografías del libro | `aerial_view.png`, `building_facade.png`, `chess.png`, `coins.png`, `futbol.png`, `smarties.png`, `starry_night.png`, `page_uneven.png`, `horse.png`, `shapes.png`, `create_shapes.py` | autor del libro | MIT, como el resto del repositorio |
| Vídeo del capítulo 16 | `853889-hd_1920_1080_25fps.mp4` | [Pexels 853889](https://www.pexels.com/video/853889/) | licencia de Pexels, la indicada en la página del vídeo (como la cita el libro) |
| Calibración | `calibration_images/` | cedidas por el autor de [niconielsen32/ComputerVision](https://github.com/niconielsen32/ComputerVision); el libro lo indica en el capítulo 14 | uso con permiso de su autor; **no** están bajo MIT |
| Marcadores y 3D | `aruco/`, `pcl_data/` | autor del libro, salvo `pcl_data/`, de los tutoriales de PCL | MIT / licencia de PCL |
| Modelos DNN | `models/` | los descargan los guiones `download_model.sh` de cada ejemplo | licencia de cada modelo (consultar su fuente) |

Notas:

- Las fotografías de la fila «Fotografías del libro» son las mismas que usan los
  guiones de `cv_book/tools` para generar las figuras, que las leen de su propia
  copia en `cv_book/images/data`. Un ejemplo lanzado sin argumentos trabaja por
  tanto sobre la misma imagen que el lector acaba de ver impresa.
- Las imágenes `right*.jpg` no aparecen en el código: `14_03_stereo_calibration`
  las deriva de las `left*.jpg` sustituyendo una palabra por la otra. Son
  necesarias aunque ninguna ruta las nombre.
- `test_pcd.pcd` lo escribe `15_05_pcl_write`, con un generador de semilla fija,
  y lo lee `15_06_pcl_read`. Está versionado para que el segundo funcione en un
  clon recién hecho.
- Los 21 ficheros de `calibration_images/` y los cinco `pcl_data/capture000*.pcd`
  se leen por comodín, no por nombre: `14_01_camera_calibration` recorre el
  directorio entero y `15_11_pcl_registration` hace lo mismo con las capturas.
  Por eso ninguna cadena del código fuente los contiene, aunque todos se usan.
- Antes de cualquier redistribución comercial conviene revisar esta tabla: si un
  fichero no aparece en ella, debe asumirse procedencia OpenCV salvo
  verificación en contra.
