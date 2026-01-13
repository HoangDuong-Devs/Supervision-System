import logging
import uuid
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams

from tracker.trackers.deepocsort.utils import *


class Qdrant:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = QdrantClient(host, port=port)

    def create(
        self,
        collection_name: str = None,
        size: int = 512,  # config
        distance: models = models.Distance.COSINE,  # config
    ):
        # check collection is exist/database connection?
        try:
            self.client.get_collection(collection_name=collection_name)
        except BaseException as e:
            if "404 (Not Found)" in str(e):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=size, distance=distance),
                )
            if "Errno 111" in str(e):
                print(f"please raise Qdrant docker host:{self.host} port:{self.port}")

    def clear_collections(
        self,
        collection_name: str = None,
    ):
        data, not_empty = self.get_data(collection_name=collection_name, all=True)
        if not_empty:
            uuids = [data.id for data in data]
            self.delete(collection_name=collection_name, PointIdsList=uuids)

    def gen_uuid(self):
        self.vector_uuid = str(uuid.uuid4())
        return self.vector_uuid

    def upsert(
        self,
        collection_name: str,
        list_data: List[dict],
        payload_struct: dict,
    ):
        """upsert data points to qdrant

        Args:
            collection_name (str): [description]
            list_data (List[dict]): [description]
            payload_struct (dict): [description]
                - Examples:
                payload_struct = {"id": None, "data": None, "status": None}
        """
        points = []
        for data in list_data:
            metadata = {key: data[key] for key in payload_struct.keys()}
            point = {
                "id": data["id"],
                "vector": data["vector"].tolist(),
                "payload": metadata,
            }
            points.append(models.PointStruct(**point))
        self.client.upsert(collection_name=collection_name, points=points)

    def count(self, collection_name: str, key: str, value: any = None):
        if key == "all":
            count = self.client.count(
                collection_name=f"{collection_name}",
                exact=True,
            )
        else:
            count = self.client.count(
                collection_name=f"{collection_name}",
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=f"{key}", match=models.MatchValue(value=f"{value}")
                        ),
                    ]
                ),
                exact=True,
            )
        return count

    def get_data(
        self,
        collection_name: str,
        with_vectors: bool = False,
        **key_value: dict,
    ):
        all = key_value.get("all", None)
        ids = key_value.get("ids", None)

        if all is not None:
            results, _ = self.client.scroll(
                collection_name=f"{collection_name}",
                # scroll_filter=models.Filter(
                #     must_not=[
                #         models.FieldCondition(key="assign", match=models.MatchValue(value=0)),
                #         ]
                #     ),
                with_payload=True,
                with_vectors=with_vectors,
                limit=100000,
            )

            return results, len(results) > 0

        elif ids is not None:
            results = self.client.retrieve(
                collection_name=f"{collection_name}",
                ids=ids,
                with_payload=True,
                with_vectors=with_vectors,
            )
            return results, len(results) > 0

        else:
            results, _ = self.client.scroll(
                collection_name=f"{collection_name}",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                        for key, value in zip(key_value.keys(), key_value.values())
                    ]
                ),
                with_payload=True,
                with_vectors=with_vectors,
                limit=100000,
            )
            return results, len(results) > 0

    def update(self, collection_name, mode, point_ids: List, key_value: dict):
        if mode == "payload":
            self.client.set_payload(
                collection_name=f"{collection_name}",
                payload=key_value,
                points=point_ids,
            )

    def delete(self, collection_name: str, PointIdsList: List):
        self.client.delete(
            collection_name=f"{collection_name}",
            points_selector=models.PointIdsList(
                points=PointIdsList,
            ),
        )

    def search(self, collection_name: str, vector: np.ndarray, limit: int, **key_value):
        results = self.client.search(
            collection_name=f"{collection_name}",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(
                            value=value,
                        ),
                    )
                    for key, value in zip(key_value.keys(), key_value.values())
                ]
            ),
            query_vector=vector.tolist(),
            # with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        results = search_result(results)
        return results

    def search_v1(
        self,
        collection_name: str,
        vector: np.ndarray,
        threshold: float = 0.75,
        limit: int = 100,
        must_not: dict = {},
        **key_value,
    ):
        results = self.client.search(
            collection_name=f"{collection_name}",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(
                            value=value,
                        ),
                    )
                    for key, value in zip(key_value.keys(), key_value.values())
                ],
                must_not=[
                    models.FieldCondition(
                        key=key, match=models.MatchValue(value=value)
                    )
                    for key, value in zip(must_not.keys(), must_not.values())
                ],
            ),
            query_vector=vector.tolist(),
            # with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        results = [result for result in results if result.score > threshold]
        return results, len(results) > 0


class search_result:
    def __init__(self, results) -> None:
        self.results = results
        self.len = len(results)
        if self.len > 0:
            self.list_idx = [point.payload["cr_count"] for point in results]
            self.majority, self.num_majority = majority_element(self.list_idx)

    def apply_threshold(self, threshold):
        results = [result for result in self.results if result.score > threshold]
        results = search_result(results)
        return results


def rematching(qdrant_client, collection_name, max_cr_count, vector, limit, init_round):
    if init_round is True:
        result = search_result([])
    else:
        result = qdrant_client.search(
            collection_name=collection_name, vector=vector, limit=limit
        )
    if result.len > 0:
        if result.majority is None:
            result = result.apply_threshold(0.7)
            if len(result.results) > 0:
                rematching_id = result.results[0].payload["cr_count"]
            else:
                rematching_id = max_cr_count + 1
                max_cr_count += 1
        else:
            if result.num_majority > 5:
                result = result.apply_threshold(0.6)
                if len(result.results) > 0:
                    rematching_id = result.majority
                else:
                    rematching_id = max_cr_count + 1
                    max_cr_count += 1

            elif result.num_majority <= 5:
                result = result.apply_threshold(0.7)
                if len(result.results) > 0:
                    rematching_id = result.results[0].payload["cr_count"]
                else:
                    rematching_id = max_cr_count + 1
                    max_cr_count += 1
    else:  # result.len ==0
        rematching_id = max_cr_count + 1
        max_cr_count += 1
    return rematching_id, max_cr_count


# ## long 
import os
import uuid
from typing import List, Literal, Tuple, Union

import numpy as np
# from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams

CastData = Union[tuple, dict, list, bytes, str, None]
# load_dotenv(find_dotenv())


class QdrantReId:
    def __init__(self, host: str = "localhost", port: int = 6444):
        self.host = host
        self.port = port
        self.client = QdrantClient(host, port=port)
        name_collection = "test"
        self.create(name_collection)
        

    def gen_uuid(self):
        return str(uuid.uuid4())

    def upsert(
        self,
        collection_name: str,
        list_data: List[dict],
        payload_struct: dict,
    ):
        """upsert data points to qdrant

        Args:
            collection_name (str): [description]
            list_data (List[dict]): [description]
            payload_struct (dict): [description]
                - Examples:
                payload_struct = {"id": None, "data": None, "status": None}
        """
        points = []
        for data in list_data:
            metadata = {key: data[key] for key in payload_struct.keys()}
            point = {
                "id": data["id"],
                "vector": data["vector"].tolist(),
                "payload": metadata,
            }
            points.append(models.PointStruct(**point))
        self.client.upsert(collection_name=collection_name, points=points)

    def update(self, collection_name, mode, point_ids: List, key_value: dict):
        if mode == "payload":
            self.client.set_payload(
                collection_name=f"{collection_name}",
                payload=key_value,
                points=point_ids,
            )

    def get_data(
        self,
        collection_name: str,
        **key_value: dict,
    ):
        if "all" in key_value.keys():
            return self.client.scroll(
                collection_name=f"{collection_name}",
                with_payload=True,
                with_vectors=True,
                limit=100000,
            )
        else:
            return self.client.scroll(
                collection_name=f"{collection_name}",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                        for key, value in zip(key_value.keys(), key_value.values())
                    ]
                ),
                with_payload=True,
                with_vectors=True,
                limit=100000,
            )

    def get_distance_type(self, collection_name: str) -> str:
        info_collection = [
            item for item in self.client.get_collection(collection_name=collection_name).config.params.vectors
        ]
        return info_collection[1][1].value

    def get_threshold(self, distance_type) -> float:
        if distance_type == "Cosine":
            return 0.78
        else:
            return 200

    def search(
        self,
        query_vector: List,
        collection_name: str = None,
        cameraID:str = None
    ):
        """
        The `search` function takes a query vector, collection name, and top-k value as input, performs
        a search using the query vector in the specified collection, and returns the name of the
        document with the highest distance score that exceeds the threshold.

        :param query_vector: The query_vector parameter is a list that represents the vectorized query
        for searching in the collection. It contains the numerical values that represent the features or
        attributes of the query
        :type query_vector: List
        :param collection_name: The `collection_name` parameter is the name of the collection in which
        you want to perform the search. A collection is a group of similar documents that are stored
        together for efficient searching and retrieval
        :type collection_name: str
        :param top_k: The `top_k` parameter specifies the maximum number of search results to return. It
        determines how many nearest neighbors to retrieve from the index
        :type top_k: int
        :return: the name of the document that matches the search query.
        """
        
        if cameraID is None:
            filter = None
        else:
            filter = models.Filter(
                must=[
                    models.FieldCondition(key="camera_id", match=models.MatchValue(value=cameraID)),
                ]
            )
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter= filter,
            limit=1,
            with_payload=True,
        )
        for result in search_result:
            distance = result.score
            distance_type = self.get_distance_type(collection_name)
            if distance >= self.get_threshold(distance_type):
                return result.payload[os.environ["QUERY_KEY"]]
        return None

    def count(self, collection_name, value):
        cnt = self.client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(key=os.environ["QUERY_KEY"], match=models.MatchValue(value=value)),
                ]
            ),
            exact=True,
        )

        return cnt.count

    def delete_points(self, collection_name: str, PointIdsList: List):
        self.client.delete(
            collection_name=f"{collection_name}",
            points_selector=models.PointIdsList(
                points=PointIdsList,
            ),
        )

    def deleted_record(self, collection_name, record_id: List[str]):
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=record_id,
            ),
        )

    def cast_filter(self, data: CastData) -> CastData:
        if isinstance(data, List):
            for item in data:
                assert isinstance(item, Tuple), "Data element must be tuple"
                assert len(item) == 2, "When filter_pare's type is list, element must be in form: Tuple(key,value)"
            return data
        elif isinstance(data, Tuple):
            temp = []
            temp.append(data)
            return temp
        else:
            # return data
            assert isinstance(data, Tuple), "data must be List or Tuple "

    def query_infor(
        self,
        collection_name,
        filter_pare: Union[Tuple, List, None] = None,
        payload_query_key: Union[str, None] = None,
        vectors_count=100000,
        with_payload: bool = True,
        with_vectors: bool = False,
        attr: Literal["id", "payload", "vector", None] = None,
    ):
        if attr == "payload" and payload_query_key is not None:
            assert (
                with_payload is not False
            ), "If attr == 'payload' or payload_query_key , param 'with_payload' need to be True"

        if payload_query_key is not None:
            assert (
                payload_query_key is not None and attr == "payload"
            ), "If attr == 'payload' or payload_query_key , param 'with_payload' need to be True"

        if attr == "vector":
            assert with_vectors is not False, "If attr == 'vector', param 'with_payload' need to be True"

        if vectors_count > 0:
            if filter_pare:
                filter_pare = self.cast_filter(filter_pare)
                filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                        for key, value in filter_pare
                    ]
                )
            else:
                filter = None
            infor, _ = self.client.scroll(
                collection_name=collection_name,
                with_payload=with_payload,
                with_vectors=with_vectors,
                limit=vectors_count,
                scroll_filter=filter,
            )

            if attr is not None:
                if attr == "payload" and payload_query_key is not None:
                    try:
                        return [getattr(record, attr)[payload_query_key] for record in infor]
                    except KeyError as e:
                        print(f"⛔ Payload doesn't have key {e}.")
                        return []

                return [getattr(record, attr) for record in infor]
            elif attr is None:
                return infor

        else:
            return []
    
    def create(
        self,
        collection_name: str = None,
        size: int = 512,  # config
        distance: models.Distance = models.Distance.COSINE,  # config
    ):
        # check collection is exist/database connection?
        try:
            self.client.get_collection(collection_name=collection_name)
        except BaseException as e:
            if "404 (Not Found)" in str(e):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=size,
                        distance=distance,
                        hnsw_config=models.HnswConfigDiff(
                            m=32,
                            ef_construct=123,
                        ),
                    quantization_config=models.ProductQuantization(
                        product=models.ProductQuantizationConfig(
                            compression=models.CompressionRatio.X32,
                            always_ram=True,
                        ),
                    ),
                    on_disk=False
                        
                    ),
                    hnsw_config=models.HnswConfigDiff(
                            m=32,
                            ef_construct=123,
                            on_disk= False
                        ),
                )
            if "Errno 111" in str(e):
                print(f"please raise Qdrant docker host:{self.host} port:{self.port}")
    
    def search_v1(
        self,
        collection_name: str,
        vector: np.ndarray,
        threshold: float = 0.6,
        limit: int = 100,
        **key_value,
    ):
        results = self.client.search(
            collection_name=f"{collection_name}",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(
                            value=value,
                        ),
                    )
                    for key, value in zip(key_value.keys(), key_value.values())
                ]
            ),
            query_vector=vector.tolist(),
            # with_vectors=True,
            with_payload=True,
            limit=limit,
        )
        results = [result for result in results if result.score > threshold]
        return results, len(results) > 0
    
    
    def get_new_data_v2(self, collection_name, camera_id, max_bbox_id=None):
        """
        The function `get_new_data_v2` retrieves new data from a Qdrant client based on specified parameters
        and updates the payload status of the retrieved data.

        :return: two values: `qdr_trks` and `max_cr_count`.
        """
        if self.client.get_collection('test').points_count > 0:
            if camera_id is not None:
                qdr_trks = self.query_infor(
                    collection_name=collection_name,
                    filter_pare=[("status", 1), ("camera_id", camera_id), ("type", "storage") ],
                    with_vectors=True,
                )
            else:
                qdr_trks = self.query_infor(
                    collection_name=collection_name,
                    filter_pare=[("status", 1), ("type", "storage")],
                    with_vectors=True,
                )

            qdr_trk_ids = [qdr_trks[i].id for i in range(len(qdr_trks))]
            qdr_bbox_ids = [qdr_trks[i].payload[os.environ["QUERY_KEY"]] for i in range(len(qdr_trks))]

            

            list_cr_count = self.query_infor(
                collection_name=collection_name,
                attr="payload",
                payload_query_key=os.environ["QUERY_KEY"],
            )

            if len(list_cr_count) > 0:
                max_cr_count = max(list_cr_count)
            else:
                max_cr_count = 0

            if len(qdr_trk_ids) > 0:
                payload = {"status": 0}
                self.update(collection_name, "payload", qdr_trk_ids, payload)
            else:
                max_bbox_id = max_bbox_id

            return qdr_trks, max_cr_count, qdr_trk_ids, qdr_bbox_ids
        
        return [], 0, [], []