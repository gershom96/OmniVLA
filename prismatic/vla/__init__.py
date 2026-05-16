def get_vla_dataset_and_collator(*args, **kwargs):
    from .materialize import get_vla_dataset_and_collator as _get_vla_dataset_and_collator

    return _get_vla_dataset_and_collator(*args, **kwargs)


__all__ = ["get_vla_dataset_and_collator"]
