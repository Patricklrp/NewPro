# exp_007 幻觉文本-图像对汇总

- 总样本数: 500
- 幻觉样本数: 106
- 幻觉比例: 21.20%

## 高频幻觉词（原词 Top 15）
- table: 9
- chair: 5
- cup: 5
- people: 5
- orange: 5
- person: 4
- car: 4
- book: 4
- bowl: 4
- bird: 3
- driver: 3
- backpack: 3
- dog: 3
- vase: 3
- sink: 3

## 高频幻觉词（映射词 Top 15）
- person: 17
- dining table: 10
- chair: 7
- bird: 7
- cup: 5
- orange: 5
- bowl: 5
- car: 4
- book: 4
- backpack: 3
- dog: 3
- vase: 3
- sink: 3
- truck: 2
- cake: 2

## 幻觉样本（按 CHAIRi 降序，Top 30）
1. image_id=277888, CHAIRi=1.000, hallu=['elephant->elephant', 'owl->bird']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000277888.jpg
   caption=The image features a stuffed hamster in the center, surrounded by a group of stuffed animals. To the left, an elephant figurine is placed closely next to the hamster, while to the right, an owl and a chicken can be seen creating a playful scene. The stuffed
2. image_id=479832, CHAIRi=1.000, hallu=['truck->truck', 'bird->bird']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000479832.jpg
   caption=The image depicts an artistically decorated vehicle, potentially a large truck, covered fully from head to toe in graffiti. Several birds can be seen on both the vehicle and the surrounding area, making the scene appear lively and colorful.

Apart from the graffiti and presence
3. image_id=456292, CHAIRi=1.000, hallu=['cup->cup', 'wine glass->wine glass']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000456292.jpg
   caption=The image displays a black, graffiti-covered wall that has been drawn on with several pictures. The artwork features an array of designs, including circles, the profile of a cup, and the letter "P." The lower right side of the wall directly next to the cup illustration shows a wine glass
4. image_id=101280, CHAIRi=1.000, hallu=['dog->dog', 'kite->kite']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000101280.jpg
   caption=The image captures a calm beach filled with the gentle sound of waves crashing against the shoreline. A dog is sitting in the waves, observing the beach as the sea gently recedes. The beach is a fitting area to be outdoors, with a few kites flying in the sky,
5. image_id=60434, CHAIRi=1.000, hallu=['driver->person']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000060434.jpg
   caption=The image shows a pole with street signs and a sign for parking removal. The pole holds a yellow street sign on top of it, and another smaller sign is below it, adding information for drivers. Additionally, there is a black no parking sign near the street signs, indicating that parking is not allowed in
6. image_id=189698, CHAIRi=1.000, hallu=['car->car']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000189698.jpg
   caption=The scene displays a peaceful street located near an intersection. A brown car is driving down the road, turning onto Bodacious Drive. The brown car is the lone vehicle in view, traveling on the empty road.

The image also features a silver car driving in another direction, even further down the
7. image_id=120388, CHAIRi=1.000, hallu=['scissors->scissors']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000120388.jpg
   caption=In the image, there is a large, grassy field with an array of openings. One such opening in the grass is placed on the left side where a piece of paper, possibly a dollar bill, is displayed. Another prominent opening in the grass appears to the left of a closed pair of scissors
8. image_id=78947, CHAIRi=1.000, hallu=['stop sign->stop sign']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000078947.jpg
   caption=The image features a stop sign attached to a metal pole, with a California street sign placed directly underneath it. The California street sign displays the number 2213 and indicates that it's located near 9100 W. California St. The stop sign is attached to a pole along with
9. image_id=8457, CHAIRi=1.000, hallu=['sheep->sheep']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000008457.jpg
   caption=The image portrays two sheep in a vast, lush green field. One sheep is situated closer to the left side of the field, while the other sheep is on the right side. They appear to be grazing on the grass in the natural open area. The field also provides a sense of tranquility
10. image_id=305343, CHAIRi=0.800, hallu=['vase->vase', 'cat->cat']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000305343.jpg
   caption=The image features two decorative ceramic vases placed on a surface, one closer to the left edge of the scene and the other towards the right edge. Each vase is adorned with animal prints, with one vase featuring a cat design and the other a bird design.

The cat v
11. image_id=365822, CHAIRi=0.750, hallu=['refrigerator->refrigerator', 'microwave->microwave']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000365822.jpg
   caption=The image shows a small, cluttered, and messy kitchen filled with various items. A refrigerator can be seen on the left side of the room, where a bottle is also placed. There is a microwave on the counter, and another microwave is mounted on the wall above
12. image_id=537611, CHAIRi=0.667, hallu=['people->person', 'person->person', 'suitcase->suitcase']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000537611.jpg
   caption=The scene features a black Harley Davidson motorcycle with various parts attached, parked next to a brick wall. There are several people visible in the image, including one person standing directly behind the motorcycle and another person on the right side. 

Additionally, there are multiple suitcases of different sizes
13. image_id=242465, CHAIRi=0.667, hallu=['people->person', 'umbrella->umbrella']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000242465.jpg
   caption=The image is of a wet street with a red stop sign prominently placed. A sign advertising a painting service stands nearby, which is also drenched with rain. Additionally, there is a group of people and an umbrella located towards the right side of the scene, possibly waiting for the rain to
14. image_id=571437, CHAIRi=0.667, hallu=['table->dining table']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000571437.jpg
   caption=The image displays a dining table with a serving of delicious fruit and vegetables. There is a ripe peeled mango placed centrally on the table, surrounded by a group of fresh ingredients. Three broccoli stalks are positioned around the mango, bringing a vibr
15. image_id=456446, CHAIRi=0.667, hallu=['surfboard->surfboard']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000456446.jpg
   caption=In the image, a man is surfing on a surfboard amidst a turbulent and choppy ocean, riding the waves as the water sprays into the air. He holds the surfboard with one hand, displaying skill and balance while standing on it. The scale of the scene
16. image_id=331329, CHAIRi=0.667, hallu=['bird->bird']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000331329.jpg
   caption=The image features a zebra standing on a lush green grassy hill amidst tall trees, bushes, and flowers. It appears majestic, enjoying the beautiful surroundings. 

Several birds are present in the scene, with four birds situated at the lower part of the image
17. image_id=63791, CHAIRi=0.667, hallu=['frisbee->frisbee']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000063791.jpg
   caption=The image features a group of young men playing a game of frisbee on a grassy field or park. They appear to be enjoying themselves, as they participate in the fun activity together. One person is standing apart from the group and holding a frisbee, while the others are scattered across
18. image_id=386962, CHAIRi=0.500, hallu=['tie->tie', 'handbag->handbag']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000386962.jpg
   caption=The image shows a red trunk next to a pair of yellow shoes. The red suitcase appears to be tied, showcasing a sense of adventure. In the vicinity of the red suitcase, a pair of two handbags - one yellow and one pink - can be seen, placed
19. image_id=236739, CHAIRi=0.500, hallu=['orange->orange', 'person->person']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000236739.jpg
   caption=The image displays a double-decker bus traveling on a large bridge with the city below. The bus is prominently in the foreground, a large orange and yellow vehicle, indicating that it could be in England. A small person can also be seen in the image, likely marveling at the impress
20. image_id=123053, CHAIRi=0.500, hallu=['parking meter->parking meter', 'people->person']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000123053.jpg
   caption=The image features a red fire hydrant positioned near the sidewalk of a busy city street. The hydrant is situated on the side of a building, and there is a parking meter right next to it. 

In the surrounding area, there are some people walking by on the sidewalk, giving
21. image_id=473124, CHAIRi=0.500, hallu=['cup->cup', 'bowl->bowl']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000473124.jpg
   caption=The image features a cranberry, orange, and yellow sauce being poured into a ramekin or a measuring cup. A sauce strainer or colander is also present in the scene, with a bowl directly below it. There's also a basket and an orange to the side,
22. image_id=488788, CHAIRi=0.500, hallu=['driver->person']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000488788.jpg
   caption=The image features a street sign displayed on a pole near a roadway, welcoming drivers to Eastcote. The street sign is green and situated close to trees. A colorful house can also be seen close to the road, while a few cars are parked or moving along the street. 

There
23. image_id=404984, CHAIRi=0.500, hallu=['potted plant->potted plant']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000404984.jpg
   caption=The image portrays a wide-open living room space, illuminated by bright sunlight. The living room features hardwood flooring and is decorated with a television placed on the left wall, hanging above the room. Other decorative elements in the living room include a potted plant, a v
24. image_id=45229, CHAIRi=0.500, hallu=['toaster->toaster']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000045229.jpg
   caption=The image features a residential kitchen with a focus on a kitchen window. Vines are climbing across the window, and a muffin and a potted plant can be seen sitting next to it. Below the window, a countertop with various appliances is visible, including a toaster, coffee pot
25. image_id=332480, CHAIRi=0.500, hallu=['oven->oven']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000332480.jpg
   caption=The image depicts a well-lit white kitchen with large windows providing ample sunlight. White appliances, including an oven and a microwave, are neatly placed in the wooden-framed space. An automatic dishwasher is situated on one side of the kitchen.
26. image_id=567787, CHAIRi=0.500, hallu=['container->bowl']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000567787.jpg
   caption=The image depicts an empty bathroom with brown wood walls and a wooden floor. A tall white toilet is positioned next to the wall, and two white wash basins are situated close to each other. To various containers are placed in the bathroom, including multiple vases and a small p
27. image_id=537982, CHAIRi=0.500, hallu=['broccoli->broccoli']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000537982.jpg
   caption=The image showcases a large collection of different types of fresh vegetables, many of which are leafy greens. Among these vegetables, there are several carrots arranged neatly, alongside greens such as lettuce and broccoli. A few batons of sweet potatoes are also present in the
28. image_id=119861, CHAIRi=0.429, hallu=['passenger->person', 'people->person', 'person->person']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000119861.jpg
   caption=The image captures a dramatic scene of passengers in route on a train. The main focus is on the motion blur of people in the train, creating a sense of the train's speed and the frozen-in-motion effect. There is one clear image of a single person on the train,
29. image_id=110783, CHAIRi=0.400, hallu=['bowl->bowl', 'sink->sink']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000110783.jpg
   caption=The image depicts a white toilet in the corner of a bathroom stall. As we look over the lid of the toilet, we can see that it is used and quite dirty, with brown streaks on the porcelain bowl. Around the toilet, the sink
30. image_id=12666, CHAIRi=0.400, hallu=['pigeon->bird']
   image=/home/ciram25-liurp/dataset/coco/val2014/COCO_val2014_000000012666.jpg
   caption=The image depicts a colorful, yellow city truck cab driving down a busy road. The truck appears to be involved in transporting excited pigeons, as indicated by the colorful words painted on its side, which read "Pigeon Travail" in a unique manner.