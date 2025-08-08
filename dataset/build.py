from dataset.imagenet import build_imagenet, build_imagenet_code



def build_dataset(dataset, data_path, datatype, transform, condition_frames, iw, ih):
    # images
    if dataset == 'imagenet':
        return build_imagenet(data_path, transform, condition_frames)
    if dataset == 'imagenet_code':
        return build_imagenet_code(data_path)

    raise ValueError(f'dataset {dataset} is not supported')