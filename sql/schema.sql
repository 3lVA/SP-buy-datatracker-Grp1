-- =====================================================
-- Create and select the database
-- =====================================================
CREATE DATABASE IF NOT EXISTS sp_buy_datatracker;
USE sp_buy_datatracker;

-- =====================================================
-- Create the customers table
-- =====================================================
CREATE TABLE IF NOT EXISTS customers (
    country_code CHAR(2) NOT NULL,
    customer_id VARCHAR(64) NOT NULL,
    mobile_verified BOOLEAN,
    num_orders_last_50days INT,
    num_cancelled_orders_last_50days INT,
    num_refund_orders_last_50days INT,
    total_payment_last_50days DECIMAL(38, 18),
    num_associated_customers INT,
    first_order_datetime DATETIME(6),
    PRIMARY KEY (country_code, customer_id)
) COLLATE = utf8mb4_0900_bin;

-- =====================================================
-- Create the orders table
-- =====================================================
CREATE TABLE IF NOT EXISTS orders (
    country_code CHAR(2) NOT NULL,
    order_id VARCHAR(64) NOT NULL,
    collect_type VARCHAR(128),
    payment_method VARCHAR(128),
    order_value DECIMAL(38, 18),
    num_items_ordered INT,
    refund_value DECIMAL(38, 18),
    order_date DATE NOT NULL,
    PRIMARY KEY (country_code, order_id)
) COLLATE = utf8mb4_0900_bin;

-- =====================================================
-- Create the labels table
-- =====================================================
CREATE TABLE IF NOT EXISTS labels (
    country_code CHAR(2) NOT NULL,
    order_id VARCHAR(64) NOT NULL,
    customer_id VARCHAR(64) NOT NULL,
    is_fraud TINYINT NOT NULL,
    PRIMARY KEY (country_code, order_id),
    CONSTRAINT labels_binary_fraud CHECK (is_fraud IN (0, 1)),
    CONSTRAINT labels_order_fk FOREIGN KEY (country_code, order_id)
        REFERENCES orders (country_code, order_id),
    CONSTRAINT labels_customer_fk FOREIGN KEY (country_code, customer_id)
        REFERENCES customers (country_code, customer_id)
) COLLATE = utf8mb4_0900_bin;
