def hd(img,gt):
    img,gt=np.atleast_1d(img.astype(np.bool_)),np.atleast_1d(gt.astype(np.bool_))
    footprint = generate_binary_structure(img.ndim, 1)
    img_border = img ^ binary_erosion(img, structure=footprint, iterations=1)
    gt_border = gt ^ binary_erosion(gt, structure=footprint, iterations=1)
    img_coords = np.argwhere(img_border)
    gt_coords = np.argwhere(gt_border)
    hd=directed_hausdorff(img_coords,gt_coords)[0]
    return hd
