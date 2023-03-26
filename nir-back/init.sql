CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

create table evaluations
(
    unique_id          bigserial
        constraint evaluations_pkey
            primary key,
    uuid               uuid                     default uuid_generate_v4()    not null,
    created_timestamp  timestamp with time zone default statement_timestamp() not null,
    is_deleted         boolean                  default false                 not null,
    eval               varchar,
    coords             varchar,
    type               varchar                  default 'image'               not null
);