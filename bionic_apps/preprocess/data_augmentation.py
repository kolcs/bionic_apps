import numpy as np


def do_augmentation(ep_data, labels, n_iter=9, include_origin=True):
    aug_epochs, aug_labels, aug_ep_ind, orig_mask = [], [], [], []

    if include_origin:
        aug_epochs.extend(ep_data)
        aug_labels.extend(labels)
        aug_ep_ind.extend(range(len(labels)))
        orig_mask.extend([True] * len(labels))

    for i, ep in enumerate(ep_data):
        ampl = ep.std()
        #  1) Setting the mean value of each channel to zero
        ep -= ep.mean(axis=-1, keepdims=True)

        for _ in range(n_iter):
            #  2) Amplify by a random number
            aug = ep * np.random.uniform(.2, 5)

            #  3) Polarity inversion
            if np.random.randint(2):
                aug *= -1

            #  4) Rotation among time dimension. HINT: not the original!!!
            if np.random.randint(2):
                aug = np.flip(aug, axis=-1)

            #  5) Add random noise
            noise = np.random.normal(0, ampl, size=aug.shape)
            aug += noise

            aug_epochs.append(aug)
            aug_labels.append(labels[i])
            aug_ep_ind.append(i)
            orig_mask.append(False)

    return aug_epochs, aug_labels, aug_ep_ind, orig_mask
