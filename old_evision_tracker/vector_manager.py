from typing import List

from configs.autocfg import cfg
from modules.qdrant.qdrant_reid import Qdrant
from tracker.utils import logger as LOGGER


class VectorManager(object):
    def __init__(
        self,
        qdrant_client: Qdrant = Qdrant(host=cfg.REID_QDRANT.HOST, port=cfg.REID_QDRANT.PORT),
        collection_name: str = None,
        **kwargs
        ):
        self.vector_uuids = dict()
        self.counters = dict()
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.to_del = []
        self.exist_id_range = kwargs.get("exist_id_range")
        self.logger = kwargs.get('logger', None)
        print(f"VectorManager parameters: host: {qdrant_client.host}| port: {qdrant_client.port}| collection_name: {self.collection_name} | exist_id_range: {self.exist_id_range}")

    def sync(self):
        """Sync the vector_uuids of manager with qdrant database when init"""
        if self.exist_id_range is not None:
            for id in range(self.exist_id_range[0], self.exist_id_range[1] + 1):
                data, not_empty = self.qdrant_client.get_data(
                    collection_name=self.collection_name, bbox_id=id
                )
                if not_empty:
                    uuids = [data.id for data in data]
                    self.vector_uuids.update({id: uuids})
                    self.counters[id] = len(self.vector_uuids[id])
                else:
                    pass
        else:
            pass

    def add_to_manager(self, uuid, key):
        """Add a new vector_uuid of bbox_id to the manager"""
        if key not in self.vector_uuids.keys():
            self.vector_uuids[key] = [uuid]
            self.counters[key] = 1
        else:
            self.vector_uuids[key].append(uuid)
            self.counters[key] = self.counters[key] + 1

    def manage_vector(self):
        self.remove_of_manager()
        self.remove_of_qdrant()
        # TODO: add more options

    def remove_of_manager(self):
        """remove the oldest vector_uuids when add the new vector_uuids"""
        position = 0
        for key, value in self.counters.items():
            if value > cfg.REID.NUM_FEATURES + 1:
                removed_flag = len(self.vector_uuids[key]) - cfg.REID.NUM_FEATURES
                self.counters[key] = self.counters[key] - removed_flag
                self.to_del = self.to_del + self.vector_uuids[key][0 : removed_flag + 1]
                del self.vector_uuids[key][0 : removed_flag + 1]
            elif value == cfg.REID.NUM_FEATURES + 1:
                self.counters[key] = self.counters[key] - 1
                self.to_del.append(self.vector_uuids[key][position])
                self.vector_uuids[key].pop(position)
            else:
                pass

    def remove_of_qdrant(self):
        if len(self.to_del) > 0:
            self.qdrant_client.delete(self.collection_name, self.to_del)
            self.to_del = []
