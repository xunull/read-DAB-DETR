# DAB-DETR

作者实现的项目地址： https://github.com/IDEA-Research/DAB-DETR

项目中有两个模型
1. DAB-DETR
2. dab-deformable-detr (一个变体)

## DAB-DETR

1. matcher.py 文件，与Deformable DETR内容相同
2. position_encoding.py 文件中多了一个PositionEmbeddingSineHW，对应于论文中的(Width & Height Modulated)
3. attention.py 文件与Conditional DETR相同，移除了对QKV三者维度相同的检测，移除了内部的全连接，这些全连接放入了transformer网络中了
4. transformer.py 主要不同的文件
5. DABDETR.py 主要不同的文件



