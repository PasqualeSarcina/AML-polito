from typing import Literal

from torch.utils.data import DataLoader


def init_dataloader(dataset_name: Literal['spair-71k', 'pf-pascal', 'pf-willow', 'ap-10k'],
                 datatype: Literal["train", "test", "val"], transform=None, num_workers=4):
    match dataset_name:
        case 'spair-71k':
            from data.spair import SPairDataset
            dataset = SPairDataset(dataset_size="large", datatype=datatype, transform=transform)
        case 'pf-pascal':
            from data.pfpascal import PFPascalDataset
            dataset = PFPascalDataset(datatype=datatype, transform=transform)
        case 'pf-willow':
            from data.pfwillow import PFWillowDataset
            dataset = PFWillowDataset(transform=transform)
        case 'ap-10k':
            from data.ap10k import AP10KDataset
            dataset = AP10KDataset(datatype=datatype, transform=transform)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def collate_single(batch_list):
        return batch_list[0]

    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=1, collate_fn=collate_single)

    return dataset, dataloader
