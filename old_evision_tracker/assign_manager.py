from typing import List

from configs.autocfg import cfg
from modules.qdrant.qdrant_reid import Qdrant


class AssignTree(object):
    def __init__(self):
        self.assigntree = dict()
    
    def create_link_node(self, child, parent):
        self.assigntree[child] = parent
    
    def search(self, child):
        if child in self.assigntree.keys():
            return self.search(self.assigntree[child])
        else:
            return child

    
class AssignIDManager(object):
    def __init__(
        self, 
        qdrant_client: Qdrant = Qdrant(host=cfg.REID_QDRANT.HOST, port=cfg.REID_QDRANT.PORT),
        collection_name: str = None,
        assignment: dict = dict(), 
        approval: dict = dict(), 
        **kwargs
        ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.local_assignment = assignment
        self.approved_assignment = approval
        self.assigntree = AssignTree()
        self.removed_keys = []
        self.tuple_suspect_swapped_idx = []
        self.ids = []
        self.logger = kwargs.get('logger', None)
        self.max_age = kwargs.get('max_age', None)
        self.coexistence_ids = []
    
    def update_attributes(self, **kwargs):
        self.tuple_suspect_swapped_idx = kwargs.get("tuple_suspect_swapped_idx", self.tuple_suspect_swapped_idx)
        self.ids = kwargs.get('ids', self.ids)
        self.coexistence_ids = kwargs.get('coexistence_ids', self.coexistence_ids)
    
    def approve_suspect_case_1(self, trackers):
        """Provide approval for reid of bigger to smaller id
        """ 
        local_assignment = self.local_assignment
        assign_list = list(local_assignment.values())
        
        for key in self.assign_idx_request:
            cr_id, expected_id, count = local_assignment[key]
            if expected_id is not None and cr_id != expected_id:
                if count > cfg.REID.SUSPECT_TIME * 1.2:
                    if cr_id > expected_id and expected_id not in self.coexistence_ids[cr_id]: #reunion conditional check
                        self.approved_assignment[key] = True
                        self.assigntree.create_link_node(cr_id, expected_id)
                        self.logger.add(f"reid {cr_id} to {expected_id} {self.assigntree}")
                        if expected_id in self.ids:
                            rm_idx = self.ids.index(expected_id)
                            self.removed_keys.append(rm_idx)
                        else: pass # TODO: ? tracker is moved
                    else: pass # TODO: have no idea!
                else: pass # TODO: not sure
            else: continue
        return trackers

    def approve_suspect_case_2(self, trackers):
        """Provide approval for swapped ids cases
        """
        
        local_assignment = self.local_assignment
        update_assignment = dict()
        for tpl in self.tuple_suspect_swapped_idx:
            if trackers[tpl[0]].suspect is True or trackers[tpl[1]].suspect is True:
                trackers[tpl[0]].update_major()
                major1, count1 = trackers[tpl[0]].suspect_data ["major"]
                
                trackers[tpl[1]].update_major()
                major2, count2 = trackers[tpl[1]].suspect_data["major"]
                
                if trackers[tpl[0]].id == major2 and trackers[tpl[1]].id == major1:
                    self.approved_assignment[tpl[0]] = True; self.approved_assignment[tpl[1]] = True
                    update_assignment= {
                        tpl[0]: (trackers[tpl[0]].id, major1, count1),
                        tpl[1]: (trackers[tpl[1]].id, major2, count2),
                    }
                    trackers[tpl[0]].reset_suspect()
                    trackers[tpl[1]].reset_suspect()
                    self.local_assignment.update(update_assignment)
                    self.tuple_suspect_swapped_idx.remove(tpl)
                else: pass # TODO ? define more cases here
                
            else:
                major1, count1 = trackers[tpl[0]].suspect_data["major"]
                major2, count2 = trackers[tpl[1]].suspect_data["major"]
                if trackers[tpl[0]].id == major2 and trackers[tpl[1]].id == major1:
                    self.approved_assignment[tpl[0]] = True; self.approved_assignment[tpl[1]] = True
                    update_assignment= {
                        [tpl[0]]: (trackers[tpl[0]].id, major1, count1),
                        [tpl[1]]: (trackers[tpl[1]].id, major2, count2)
                    }
                    self.local_assignment.update(update_assignment)
                    self.tuple_suspect_swapped_idx.remove(tpl)
                else:
                    self.tuple_suspect_swapped_idx.remove(tpl)
                trackers[tpl[0]].reset_suspect()
                trackers[tpl[1]].reset_suspect()
        return trackers

    def approve_suspect_case_3(self, trackers):
        """Provide approval for reid of bigger to smaller id
        """
        temp = []
        local_assignment = self.local_assignment
        keys = []
        
        for key in self.assign_idx_request:
            cr_id, expected_id, count = local_assignment[key]
            if expected_id is not None and cr_id != expected_id:
                if count > cfg.REID.SUSPECT_TIME * 2:
                    self.approved_assignment[key] = True
                    keys.append(key)
                    if expected_id in self.ids:
                        rm_idx = self.ids.index(expected_id)
                        self.removed_keys.append(rm_idx)
                    else: pass # TODO: have no idea!
                else: pass # TODO: not sure
            else: continue
            
        for tup in self.tuple_suspect_swapped_idx:
            if not any(elem in tup for elem in keys):
                temp.append(tup)
                
        self.tuple_suspect_swapped_idx = temp
        return trackers
            
    def update_expected_assignment_to_root(self):
        """Result of searchs for expected assignment cannot be expected root id.
        This function queries and update expected assignment to root id.
        Root id should be assigned at the previous assign.
        """
        dict_update_trk_id = {}
        self.swapped_trk_idxs = [item for sublist in self.tuple_suspect_swapped_idx for item in sublist]
        for key, value in self.approved_assignment.items():
            if (
                key in self.local_assignment 
                and key not in self.swapped_trk_idxs
                and value is True
                ):
                root_id = self.assigntree.search(self.local_assignment[key][0])
                # if self.local_assignment[key][1] != root_id and root_id not in self.coexistence_ids[self.local_assignment[key][1]] and root_id not in self.cr_trk_ids: 
                if self.local_assignment[key][1] != root_id and root_id not in self.cr_trk_ids: 
                    dict_update_trk_id.update({key: (self.local_assignment[key][0], root_id)})
                    self.local_assignment.update(dict_update_trk_id)       
                # else:
                
    def assign_to_root(self, trackers):
        dict_update_trk_id = dict()
        for key, value in self.local_assignment.items():
            root_id = self.assigntree.search(self.local_assignment[key][0])
            if self.local_assignment[key][0] != root_id:
                if root_id in self.ids:
                    idx = self.ids.index(root_id)
                else:
                    idx = key
                if key != idx:
                    trackers[key].id = root_id
                    # self.removed_keys.append(idx)
                    dict_update_trk_id.update({key: (root_id, None, 0)})
                else:
                    trackers[key].id = root_id
                    dict_update_trk_id.update({key: (root_id, None, 0)})
                data, check = self.qdrant_client.get_data(collection_name=self.collection_name, bbox_id = self.local_assignment[key][0])
                if check: # reunion
                    uuids = [data.id for data in data]
                    payload = {
                        "bbox_id": root_id
                    }
                    self.qdrant_client.update(self.collection_name, 'payload', uuids, payload)
        self.local_assignment.update(dict_update_trk_id)
        return trackers
    
    def assign_id(self, trackers):
        """Assign expected old id for current id of tracker based on approved assignment.
        Approved assignment like triggers that allows current id of tracker 
        turn back to the nearest old id.
        """
        dict_update_trk_id = {}
        for key, value in self.approved_assignment.items():
            if key in self.local_assignment:
                if value is True:
                    trackers[key].id = self.local_assignment[key][1]
                    trackers[key].reset_suspect(False)
                    # dict_update_trk_id.update({key: (self.local_assignment[key][1],self.local_assignment[key][1])})
                    self.approved_assignment[key] = False
        self.local_assignment.update(dict_update_trk_id)
        return trackers

    def update_trk_attributes(self, trackers):
        dict_update_trk_id = {}
        for key, value in self.approved_assignment.items():
            try: 
                suspect = trackers[key].suspect 
                if suspect is False and self.approved_assignment[key] is False:
                    dict_update_trk_id.update({key: (self.local_assignment[key][0], None, 0)})
            except:
                print(f"key {key} not in list, length of tracker list is {len(trackers)}")
            
            #will dead
            if key in self.removed_keys:
                self.logger.add(f"idx is needed to removed {key}")
                trackers[key].time_since_update = self.max_age + 10
                
        self.removed_keys = []
        self.local_assignment.update(dict_update_trk_id)
        return trackers
        