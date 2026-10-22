"""Buffers whose shape is only known once data has been seen."""

from typing import Tuple


class LazyBuffersMixin:
    """Mixin for modules that create some of their buffers from the first data.

    Subclasses register each such buffer as ``None`` and list its name in
    ``_lazy_buffers``. A ``None`` buffer is left out of ``state_dict()``, so
    loading needs help in both directions, which ``_load_from_state_dict``
    provides:

    - an entry for a buffer that does not exist yet creates it, so a freshly
      built module can load the state of a trained one;
    - a missing entry for a buffer that exists resets it to ``None``, so
      restoring a state saved before the buffer was created gives back that
      state.

    ``initialized`` is True once every lazy buffer exists, whether it was
    created by an update or by loading.
    """

    _lazy_buffers: Tuple[str, ...] = ()

    @property
    def initialized(self) -> bool:
        return all(self._buffers.get(name) is not None for name in self._lazy_buffers)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in self._lazy_buffers:
            key = prefix + name
            if key in state_dict:
                if self._buffers.get(name) is None:
                    self._buffers[name] = state_dict[key].detach().clone()
            else:
                self._buffers[name] = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
