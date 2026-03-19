from abc import abstractmethod


class Router[S]:
    @abstractmethod
    def search(self, root_state: S) -> list[int]:
        pass
