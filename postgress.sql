CREATE TABLE "categories" (
  "id" SERIAL PRIMARY KEY,
  "name" varchar NOT NULL UNIQUE,
  "created_at" timestamp DEFAULT now()
);

CREATE TABLE "product_categories" (
  "product_id" integer,
  "categorie_id" integer,
  "created_at" timestamp DEFAULT now(),
  PRIMARY KEY ("product_id", "categorie_id")
);

CREATE TABLE "products" (
  "id" SERIAL PRIMARY KEY,
  "title" varchar NOT NULL,
  "description" text,
  "price" float,
  "date_first_available" timestamp,
  "main_category_id" integer,
  "created_at" timestamp DEFAULT now()
);

CREATE TABLE "ratings" (
  "product_id" integer,
  "value" float,
  "count" integer,
  "created_at" timestamp DEFAULT now()
);

CREATE TABLE "images" (
  "id" SERIAL PRIMARY KEY,
  "product_id" integer,
  "path" varchar,
  "created_at" timestamp DEFAULT now()
);

ALTER TABLE "product_categories" ADD FOREIGN KEY ("product_id") REFERENCES "products" ("id") DEFERRABLE INITIALLY IMMEDIATE;
ALTER TABLE "product_categories" ADD FOREIGN KEY ("categorie_id") REFERENCES "categories" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "products" ADD FOREIGN KEY ("main_category_id") REFERENCES "categories" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "ratings" ADD FOREIGN KEY ("product_id") REFERENCES "products" ("id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "images" ADD FOREIGN KEY ("product_id") REFERENCES "products" ("id") DEFERRABLE INITIALLY IMMEDIATE;
