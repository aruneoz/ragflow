-- Table: public.neumai_vector_test

-- DROP TABLE IF EXISTS public.neumai_vector_test;

CREATE TABLE IF NOT EXISTS public.<table_name>
(
    id character varying(100) COLLATE pg_catalog."default",
    file_id text COLLATE pg_catalog."default",
    chunk_content text COLLATE pg_catalog."default",
    chunk_metadata text COLLATE pg_catalog."default",
    embedding vector(768)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.<table_name>
    OWNER to postgres;
-- Index: neumai_vector_test_embedding_idx

-- DROP INDEX IF EXISTS public.neumai_vector_test_embedding_idx;

CREATE INDEX IF NOT EXISTS <table_name>_vector_<db_name>_embedding_idx
    ON public.sabre_vector_synxis USING ivfflat
    (embedding vector_cosine_ops)
    WITH (lists=100)
    TABLESPACE pg_default;