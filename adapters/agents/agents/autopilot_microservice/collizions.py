import weakref

class CollisionHandler:
    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.is_collision = True
