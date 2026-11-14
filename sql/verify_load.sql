USE sp_buy_datatracker;

-- =====================================================
-- Compare table counts after the first load and replay
-- =====================================================
SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM customers
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'labels', COUNT(*) FROM labels;

-- =====================================================
-- Check that labels match both parent tables
-- =====================================================
SELECT COUNT(*) AS labels_with_both_parents
FROM labels AS l
JOIN orders AS o ON o.country_code = l.country_code AND o.order_id = l.order_id
JOIN customers AS c ON c.country_code = l.country_code AND c.customer_id = l.customer_id;

-- =====================================================
-- Check fraud and non-fraud counts
-- =====================================================
SELECT is_fraud, COUNT(*) AS label_count
FROM labels
GROUP BY is_fraud
ORDER BY is_fraud;
