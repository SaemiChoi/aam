from typing import List, Generic, TypeVar

ModuleType = TypeVar('ModuleType')
ModuleListType = TypeVar('ModuleListType', bound=List)


class ObjectHooker(Generic[ModuleType]):
    def __init__(self, module: ModuleType):
        self.module: ModuleType = module

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, type, value, trace_back):
        self.unhook()

    def hook(self):
        self._hook_impl()
        return self

    def unhook(self):
        self._unhook_impl()
        return self

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        pass


class AggregateHooker(ObjectHooker[ModuleListType]):
    def _hook_impl(self):
        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        for h in self.module:
            h.unhook()





