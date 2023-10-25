
The background and contrast of the DHM image for neurite prediction are automatically set in 'Fiji,' as demonstrated in the 'Img2map_pipeline.ipynb' notebook.

For cell body segmentation, there is no need to preprocess DHM images. The images should be in 32-bit float grayscale format. The patch maker's function was developed by Maxime Moreaud, and you can find more details in this article:
[Adding geodesic information and stochastic patch-wise image prediction for small dataset learning](https://www.sciencedirect.com/science/article/pii/S092523122100196X).
