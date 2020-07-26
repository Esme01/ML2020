# Conditional Generation

​		Conditional Generation的意义是可以控制输出的内容，比如输入文字产生图片。

## Text to Image

​		传统的监督学习方法：输入一段文字，输出一张图片。训练集是文本+图片。由于训练数据中一段文字可能对应多张图片，比如“Train”,所以传统的神经网络会给出一个模糊的图片，因为是与“Train”对应的多张图片的平均。

![image-20200726142240757](C:\Users\1ceee\AppData\Roaming\Typora\typora-user-images\image-20200726142240757.png)

​		而Conditional GAN来训练模型时，生成器的输入不仅仅是一段文本，还包括Normal Distribution的噪声z。（个人感觉是原始的GAN生成仅仅取决于z，而cGAN的conditional就体现在输入要加文本（可以看作是label）,要生成什么样的图片，就告诉生成器想要的对应文本)

​		同时原始的GAN当中判别器仅输入一张图片，图片质量高（像真的）就会给出高分，而现在需要输入对应的文本和生成的图片。判别器检查图片好坏的任务在这里有两个：1.图片是否高质量；2.图片和文本能否对应。所以现在判别器给低分的情况包括两种：1.正确的文字+较差的的生成图片；2.较为真实的图片+错误的文字。

![image-20200726144204288](C:\Users\1ceee\AppData\Roaming\Typora\typora-user-images\image-20200726144204288.png)

![image-20200726144444809](C:\Users\1ceee\AppData\Roaming\Typora\typora-user-images\image-20200726144444809.png)

		## Conditional GAN ——Discriminator

​		![image-20200726144753034](C:\Users\1ceee\AppData\Roaming\Typora\typora-user-images\image-20200726144753034.png)

​		上图中第二种架构的优势是可以分清楚到底是哪一种问题（不match或者结果差）。

## Stack GAN

​		将架构分成两部分：第一部分先生成较小的图片，第二部分根据生成的小图和embedding产生大图。

![image-20200726145313560](C:\Users\1ceee\AppData\Roaming\Typora\typora-user-images\image-20200726145313560.png)

## Image to Image

​		传统的神经网络产生的图片依然比较模糊，GAN的问题是会产生奇怪的东西。让生成器在考虑“骗过”判别器的同时考虑不要与原图差别太大，得到的结果就会相对较好。

![image-20200726145653405](C:\Users\1ceee\AppData\Roaming\Typora\typora-user-images\image-20200726145653405.png)

​		如果直接输入较大的图片，网络的参数过多，训练过程很容易Overfitting或者用时过长。所以在判别过程，判别器仅检测一小部分图片，具体的大小是一个超参数。（patch GAN）

![image-20200726150004334](C:\Users\1ceee\AppData\Roaming\Typora\typora-user-images\image-20200726150004334.png)

## Other Applications

​		Speech Enhancement,Video Generation……





​		

​		



​		



​		

