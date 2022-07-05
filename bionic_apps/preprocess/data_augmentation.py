import numpy as np


def do_augmentation(ep_data, labels, n_iter=9, include_origin=True):
    aug_epochs, aug_labels, aug_ep_ind, orig_mask = [], [], [], []
    ep_ind = np.arange(len(labels))

    def add_aug(data, orig=False):
        aug_epochs.append(data)
        aug_labels.extend(labels)
        aug_ep_ind.extend(ep_ind)
        orig_mask.extend([orig] * len(labels))

    if include_origin:
        add_aug(ep_data, orig=True)

    ampl = ep_data.std()
    #  1) Setting the mean value of each channel to zero
    ep_data -= ep_data.mean(axis=-1, keepdims=True)

    for i in range(n_iter):
        #  2) Amplify by a random number
        aug = ep_data * np.random.uniform(.2, 5)

        #  3) Polarity inversion
        if np.random.randint(2):
            aug *= -1

        #  4) Rotation among time dimension. HINT: not the original!!!
        if np.random.randint(2):
            aug = np.flip(aug, axis=-1)

        #  5) Add random noise
        noise = np.random.normal(0, ampl, size=aug.shape)
        aug += noise

        add_aug(aug)

    aug_epochs = np.vstack(aug_epochs)
    return aug_epochs, aug_labels, aug_ep_ind, orig_mask
