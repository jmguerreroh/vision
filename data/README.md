# Procedencia y licencias de los datos

Los programas de ejemplo leen sus imágenes y vídeos de esta carpeta. **Aquí solo
está lo que usa algún ejemplo**: el material que dejó de usarse se conserva en
`old_data/`, en la raíz del repositorio, con la misma procedencia y licencia que
tenía. La mayor parte procede de colecciones públicas estándar del área; el
resto es material propio. El repositorio se distribuye bajo MIT, pero **esa
licencia cubre el código, no estos datos**: cada conjunto conserva la licencia de
su fuente original.

| Conjunto | Ficheros | Fuente | Licencia / condiciones |
|---|---|---|---|
| Muestras de OpenCV | `vtest.avi`, `Megamind.avi`, `messi5.jpg`, `digits.png`, `form.jpg`, `scanned-form.jpg`, `fruits.jpg`, `star.jpg`, `RGB.jpg`, `left*.jpg`, `right*.jpg`, `left.jpg`, `right.jpg`, `parasaurolophus_*.ply` | [opencv/opencv_extra](https://github.com/opencv/opencv) (carpetas `samples/data`) | Apache 2.0 |
| Par estéreo Aloe | `aloeL.jpg`, `aloeR.jpg`, `aloeGT.png` | Middlebury Stereo 2006 (Scharstein y Pal, CVPR 2007) | uso académico con cita de la fuente |
| Fotografías del libro | `aerial_view.png`, `building_facade.png`, `chess.png`, `coins.png`, `futbol.png`, `smarties.png`, `starry_night.png`, `page_uneven.png`, `horse.png`, `shapes.png`, `create_shapes.py` | autor del libro | MIT, como el resto del repositorio |
| Vídeo del capítulo 12 | `853889-hd_1920_1080_25fps.mp4` | [Pexels 853889](https://www.pexels.com/video/853889/) | Creative Commons Zero |
| Calibración y 3D | `calibration_images/`, `aruco/`, `pcl_data/` | autor del libro, salvo `pcl_data/`, de los tutoriales de PCL | MIT / licencia de PCL |
| Modelos DNN | `models/` | los descargan los guiones `download_model.sh` de cada ejemplo | licencia de cada modelo (consultar su fuente) |

Notas:

- Las fotografías de la fila «Fotografías del libro» son las mismas que usan los
  guiones de `cv_book/tools` para generar las figuras, que las leen de su propia
  copia en `cv_book/images/data`. Un ejemplo lanzado sin argumentos trabaja por
  tanto sobre la misma imagen que el lector acaba de ver impresa.
- Las imágenes `right*.jpg` no aparecen en el código: `10_03_stereo_calibration`
  las deriva de las `left*.jpg` sustituyendo una palabra por la otra. Son
  necesarias aunque ninguna ruta las nombre.
- `test_pcd.pcd` lo escribe `11_05_pcl_write`, con un generador de semilla fija,
  y lo lee `11_06_pcl_read`. Está versionado para que el segundo funcione en un
  clon recién hecho.
- Antes de cualquier redistribución comercial conviene revisar esta tabla: si un
  fichero no aparece en ella, debe asumirse procedencia OpenCV salvo
  verificación en contra.
