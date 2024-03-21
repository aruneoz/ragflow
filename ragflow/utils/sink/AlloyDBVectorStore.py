"""
@Time    : 2024/01/09 17:45
@Author  : asanthan
@Descriptor: This is a Demonstration of Distributed RAG Pipeline to process any doc , any layout including multimodal LLM - CloudSQL pgVector Sink

"""


import json
from typing import List, Optional

import numpy as np
from neumai.SinkConnectors.SinkConnector import SinkConnector
from neumai.Shared.NeumSinkInfo import NeumSinkInfo
from neumai.Shared.NeumVector import NeumVector
from neumai.Shared.NeumSearch import NeumSearchResult
import sqlalchemy
import psycopg2
from psycopg2 import pool
from psycopg2 import Error
from psycopg2.extras import Json, DictCursor
from unstructured.cleaners.core import clean_non_ascii_chars
from neumai.SinkConnectors.filter_utils import FilterCondition, FilterOperator
from pydantic import Field
import vecs

class AlloyDBConnectionException(Exception):
    """Raised if establishing a connection to AlloyDB fails"""
    pass

class AlloyDBInsertionException(Exception):
    """Raised if inserting embedding  to AlloyDB fails"""
    pass

class AlloyDBIndexInfoException(Exception):
    """Raised if index table not found in AlloyDB"""
    pass

class AlloyDBQueryException(Exception):
    """Raised if querying embedding  to AlloyDB fails"""
    pass

class AlloyDBSink(SinkConnector):
    """
    AlloyDB Sink

    A connector designed for exporting data to AlloyDB, a cloud-based database platform. It manages connections and data transfers to specific AlloyDB databases.

    Attributes:
    -----------
    database_connection : str
        Connection string or details required to connect to the AlloyDB database.

    database_user : str
        User required to connect to the AlloyDB database.

    database_pwd : str
        Pwd required to connect to the AlloyDB database.

    database_name : str
        Database name within AlloyDb where the data will be stored.

    database_table_name : str
        Database table name within AlloyDb where the data will be stored.

    """

    database_host: str = Field(..., description="Database connection for AlloyDB.")
    database_port: str = Field(..., description="Database connection for AlloyDB.")
    database_user: str = Field(..., description="Database User  for AlloyDB.")
    database_pwd: str = Field(..., description="Database Pass for AlloyDB.")
    database_name: str = Field(..., description="Database Name for AlloyDB.")
    database_table_name: str = Field(..., description="Database Table Name connection for AlloyDB.")



    @property
    def sink_name(self) -> str:
        return 'AlloyDBSink'

    @property
    def required_properties(self) -> List[str]:
        return ['database_host', 'database_port','database_user' , 'database_pwd' , 'database_name' ,'database_table_name']

    @property
    def optional_properties(self) -> List[str]:
        return []

    def getpool(self):
        print(f"Going to get connection from {self.database_host} , ")
        threaded_postgreSQL_pool = psycopg2.pool.ThreadedConnectionPool(5, 20, user=self.database_user,
                                                                        password=self.database_pwd,
                                                                        host=self.database_host,
                                                                        port=self.database_port,
                                                                        database=self.database_name)

        if (threaded_postgreSQL_pool):
            print("Connection pool created successfully using ThreadedConnectionPool")

        #ps_connection = self.threaded_postgreSQL_pool.getconn()
        return threaded_postgreSQL_pool

    def validation(self) -> bool:
        """config_validation connector setup"""

        try:
            # Create a cursor to perform database operations
            threaded_postgreSQL_pool = self.getpool()
            connection = threaded_postgreSQL_pool.getconn()
            cursor = connection.cursor()
            # Print PostgreSQL details
            print("PostgreSQL server information")
            print(connection.get_dsn_parameters(), "\n")
            # Executing a SQL query
            cursor.execute("SELECT version();")
            # Fetch result
            record = cursor.fetchone()
            print("You are connected to - ", record, "\n")
            cursor.close()
            threaded_postgreSQL_pool.putconn(connection)
            print("Put away a PostgreSQL connection")

        except Exception as e:
            raise AlloyDBConnectionException(f"AlloyDB connection couldn't be initialized. See exception: {e}")
        finally:
            if (threaded_postgreSQL_pool):


                threaded_postgreSQL_pool.closeall
                print("Threaded PostgreSQL connection pool is closed")

        return True

    def delete_vectors_with_file_id(self, file_id: str) -> bool:
        threaded_postgreSQL_pool = self.getpool()
        connection = threaded_postgreSQL_pool.getconn()
        cursor = connection.cursor()
        try:
            # insert statement
            delete_stmt = sqlalchemy.text(
                f"DELETE FROM {self.database_table_name} where file_id={file_id}",
            )

            cursor.execute(delete_stmt)
            connection.commit()
            cursor.close()
            threaded_postgreSQL_pool.putconn(connection)

        except Exception as e:
            raise Exception(f"AlloyDB deletion failed. Exception {e}")
        finally:
            if (threaded_postgreSQL_pool):
                threaded_postgreSQL_pool.closeall
                print("Threaded PostgreSQL connection pool is closed")
        return True

    def store(self, vectors_to_store: List[NeumVector]) -> int:
        threaded_postgreSQL_pool = self.getpool()
        connection = threaded_postgreSQL_pool.getconn()
        cursor = connection.cursor(cursor_factory=DictCursor)
        try:

            ## Create table if not exists

            create_table = f"""CREATE TABLE IF NOT EXISTS {self.database_table_name} (
                              id TEXT,
                              file_id TEXT,
                              chunk_content TEXT,
                              chunk_metadata TEXT,
                              embedding vector(768) 
                            );"""


            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector");
            cursor.execute(create_table)
            connection.commit()

            values = []
            for i in range(0, len(vectors_to_store)):
                metadata = json.dumps(vectors_to_store[i].metadata)
                # upsert_embeddings = f"""INSERT INTO {self.database_table_name} (id,file_id,chunk_content,chunk_metadata,embedding)
                #                                   VALUES('{vectors_to_store[i].id}','{vectors_to_store[i].metadata["filename"]}','{vectors_to_store[i].metadata["text"]}','{metadata}','{vectors_to_store[i].vector}')
                #                                  ;"""

                values.append((f'{vectors_to_store[i].id}',f'{vectors_to_store[i].metadata["filename"]}',f'{vectors_to_store[i].metadata["text"]}',f'{metadata}',f'{vectors_to_store[i].vector}'))

                #print(upsert_embeddings)
            cursor.executemany(f"""INSERT INTO {self.database_table_name} (id,file_id,chunk_content,chunk_metadata,embedding)
                                                  VALUES(%s,%s,%s,%s,%s)""", values)
            connection.commit()

            cursor.close()
            threaded_postgreSQL_pool.putconn(connection)

        except Exception as e:
            raise AlloyDBInsertionException(f"AlloyDB storing failed. Exception {e}")
        finally:
            if (threaded_postgreSQL_pool):
                threaded_postgreSQL_pool.closeall
                print("Threaded PostgreSQL connection pool is closed")
        return len(vectors_to_store)

    # def translate_to_supabase(filter_conditions: List[FilterCondition]):
    #     query_parts = []
    #
    #     for condition in filter_conditions:
    #         mongo_operator = {
    #             FilterOperator.EQUAL: '$eq',
    #             FilterOperator.NOT_EQUAL: '$ne',
    #             FilterOperator.GREATER_THAN: '$gt',
    #             FilterOperator.GREATER_THAN_OR_EQUAL: '$gte',
    #             FilterOperator.LESS_THAN: '$lt',
    #             FilterOperator.LESS_THAN_OR_EQUAL: '$lte',
    #             FilterOperator.IN: '$in',
    #         }.get(condition.operator, None)
    #
    #         if mongo_operator:
    #             query_parts.append({condition.field: {mongo_operator: condition.value}})
    #         else:
    #             # Handle complex cases like IN, NOT IN, etc.
    #             pass
    #
    #     return {"$and": query_parts}  # Combine using $and, can be changed to $or if needed

    def search(self, vector: List[float], number_of_results: int, filters: List[FilterCondition] = []) -> List:
        threaded_postgreSQL_pool = self.getpool()
        connection = threaded_postgreSQL_pool.getconn()
        cursor = connection.cursor()

        # filters = self.translate_to_supabase(filter)
        print(filters)

        try:
            query_statement = f"SELECT id , chunk_content , chunk_metadata , 1- ('{vector}' <-> embedding) AS cosine_similarity FROM {self.database_table_name} ORDER BY embedding <-> '{vector}' LIMIT {number_of_results};"
            print(query_statement)
            cursor.execute(query_statement)
            results = cursor.fetchall()
            print(results[0])
            cursor.close()
            threaded_postgreSQL_pool.putconn(connection)

        except Exception as e:
            raise AlloyDBQueryException(f"Error querying vectors from Supabase. Exception: {e}")
        finally:
            if (threaded_postgreSQL_pool):
                threaded_postgreSQL_pool.closeall
                print("Threaded PostgreSQL connection pool is closed")
        matches = []
        for result in results:
            matches.append(NeumSearchResult(
                id=str(result[0]),
                metadata=json.loads(result[2]),
                score=result[3]
            ))

        return matches

    def info(self) -> NeumSinkInfo:
        threaded_postgreSQL_pool = self.getpool()
        connection = threaded_postgreSQL_pool.getconn()
        cursor = connection.cursor()
        threaded_postgreSQL_pool.putconn(connection)
        try:
            query = f"""SELECT COUNT(*) FROM {self.database_table_name};"""

            result = cursor.execute(sqlalchemy.text(query)).fetchall()
            cursor.close()
        except:
            raise AlloyDBIndexInfoException(f"Table {self.database_table_name} does not exist")
        finally:
            if (threaded_postgreSQL_pool):
                threaded_postgreSQL_pool.closeall
                print("Threaded PostgreSQL connection pool is closed")

        return NeumSinkInfo(number_vectors_stored=result[0])