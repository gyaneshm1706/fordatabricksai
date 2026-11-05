# Genie space recipe (Lab 4)

## Space settings
- **Title:** `Sales - <your name>`
- **Warehouse:** the course SQL warehouse
- **Data (Part A):** `course.<you>_support.orders`, `customers`, `products`, `order_lines`
- **Data (Part B):** add `course.<you>_support.order_metrics`

## General instructions (paste into the space, Part B)
```
Revenue always excludes cancelled orders; prefer the order_metrics metric view for revenue, orders, AOV and late rate.
Amounts are in INR. Quarters are fiscal quarters; the fiscal year starts on 1 April.
If the user does not give a time period for a trend or total, ask which period they mean.
This space covers orders, customers, products and deliveries only. For HR, finance budgets or employees,
say that the data is not available in this space.
```

## Five sample questions (Part A: ask before semantics, note what's wrong)
1. What was revenue by region last quarter?
2. How many orders were late last week in the North region?
3. Which product category has the highest average order value?
4. Who are our top 5 gold customers by revenue this year?
5. What is our employee attrition rate? *(out of scope: should be declined)*

## Example SQL to add (Part C, when a benchmark fails)
Title: *Late orders by region, last N days*
```sql
SELECT region, COUNT(*) AS late_orders
FROM course.<you>_support.orders
WHERE eta > promised_date AND status <> 'cancelled'
  AND order_date >= current_date() - INTERVAL 7 DAYS
GROUP BY region ORDER BY late_orders DESC
```
