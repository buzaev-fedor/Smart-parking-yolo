# Smart-parking-yolo
Программа обработки RTSP потока для анализа свободных и занятых парковочных мест. 
Основано на Yolo-V4.
![alt text](https://github.com/buzaev-fedor/Smart-parking-yolo/blob/main/github.png)
## Принцип действия
1. Чертеж парковки - оператор расчерчивает парковочные места для парковки. 
2. После задания всех парковочных мест, программа в автоматическом режиме и единоразово рассчитывает соседние места (соседей) парковочных мест. И на выходе мы получаем словарь, где ключ - это айди парковочного места, а значения - соседи. Расчет проходит по такому принципу: 
Берутся координаты по x и y, а затем рассчитывается площадь парковочного места и рассчитывается метрика IoU (пересечение на объедение) с другими местами, и если оно больше нуля и меньше единицы, то добавляется в значение ключа айди соседа.
3. Потом создается RTree, которое облегчит потом расчет занятых/свободных/плохих мест. Принцип такой:
Берется координата псевдоцентра парковочного места и она сопоставляется координатам фрейма.
(Берется псевдоцентр, а не центр, потому что парковочные места из-за ракурса камеры не правильной формы)
4. Для стабильной картинки в реальном времени, я делаю так: Берется один кадр, кадр обрабатывается и отдается в ОЗУ. Кадр показывается 6 раз, а затем берется следующий кадр.
5. Берется из потока фрейм, обрезается до площади парковки и отдается нейросети, которая на выход отдает bounding box объектов. 
(Специально только часть фрейма, потому что снижается нагрузка и меньше бесполезной информации)
6. Для того, чтобы расчитать правильность парковки, берутся координаты bounding box'a по x и y и уменьшаются до 0.7 от своей изначальной, находится площадь по измененным координатам.
7. Сопоставляются координаты объекта с RTree (вместо перебора по циклу по всем парковочным местам) и рассчитывается метрика IoU. 
Если метрика меньше 0.05, то место свободно и закрашивается зеленым. Если занято, то пропадает закраска.
8. Затем проверяется метрика IoU для соседних мест, и если метрика не равна нулю, то айди плохого места добавляется  в список плохих мест.
9. Фильтруется список плохих мест, чтобы не попались хорошие места в плохих, затем плохие места раскрашиваются красным цветом.

"Плохие места" - это места, которые заняты одной машиной одновременно.

## Что еще хочется сделать:
1. Дообучить модель на отличных данных.
2. Перевести в TensorRT.
3. Сделать Deepstream.

## Reference:
    https://github.com/Tianxiaomo/pytorch-YOLOv4
    https://github.com/AlexeyAB/darknet
