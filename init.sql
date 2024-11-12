-- Drop existing tables if they exist
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Create enum type for order status
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'orderstatus') THEN
        CREATE TYPE orderstatus AS ENUM (
            'pending',
            'processing',
            'confirmed',
            'shipped',
            'delivered',
            'cancelled'
        );
    END IF;
END$$;

-- Create customers table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    default_shipping_address VARCHAR(255),
    phone VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    price DECIMAL(10,2) NOT NULL,
    inventory_count INTEGER DEFAULT 0,
    min_stock_level INTEGER DEFAULT 5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    shipping_address VARCHAR(255),
    total_amount DECIMAL(10,2),
    status orderstatus DEFAULT 'pending',
    special_instructions VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create order_items table
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Seed customers
INSERT INTO customers (name, email, default_shipping_address, phone) VALUES
    ('Acme Corporation', 'contact@acme.com', '123 Business Drive, Corporate Park, NY 10001', '(555) 123-4567'),
    ('TechStart Inc', 'orders@techstart.com', '456 Innovation Avenue, Tech Valley, CA 94016', '(555) 234-5678'),
    ('Global Solutions Ltd', 'procurement@globalsolutions.com', '789 Enterprise Road, Business District, TX 75001', '(555) 345-6789'),
    ('Data Systems Co', 'purchasing@datasystems.co', '321 Digital Lane, Tech Hub, WA 98101', '(555) 456-7890'),
    ('Smart Electronics', 'orders@smartelectronics.com', '654 Circuit Avenue, Electronics Park, CA 95014', '(555) 567-8901');

-- Seed products
INSERT INTO products (name, description, price, inventory_count, min_stock_level) VALUES
    ('Enterprise Server Pro', 'High-performance server with redundant power supply', 2499.99, 50, 10),
    ('Business Laptop Elite', '14-inch lightweight laptop for business professionals', 1299.99, 100, 20),
    ('Network Switch 24-Port', 'Managed 24-port gigabit network switch', 399.99, 75, 15),
    ('Professional Workstation', 'High-end workstation for demanding applications', 3499.99, 30, 5),
    ('Secure NAS Storage', 'Network attached storage with RAID support', 899.99, 40, 8),
    ('Business Tablet Pro', '10-inch tablet with stylus support', 699.99, 60, 12),
    ('Office Dock Station', 'Universal docking station for laptops', 249.99, 80, 15),
    ('Enterprise Router', 'Advanced router with firewall capabilities', 799.99, 45, 10);

-- Seed orders
INSERT INTO orders (customer_id, shipping_address, total_amount, status, special_instructions) VALUES
    (1, '123 Business Drive, Corporate Park, NY 10001', 4999.98, 'delivered', 'Please deliver during business hours'),
    (2, '456 Innovation Avenue, Tech Valley, CA 94016', 2599.98, 'processing', 'Require signature upon delivery'),
    (3, 'Custom Address, Suite 100, Business Center, NY 10002', 1599.98, 'pending', NULL);

-- Seed order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 2, 2499.99),  -- 2 Enterprise Server Pro for Order 1
    (2, 2, 2, 1299.99),  -- 2 Business Laptop Elite for Order 2
    (3, 3, 4, 399.99);   -- 4 Network Switch 24-Port for Order 3

-- Create indexes for better performance
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at
    BEFORE UPDATE ON products
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add some useful views
CREATE OR REPLACE VIEW order_summary AS
SELECT
    o.id AS order_id,
    c.name AS customer_name,
    o.shipping_address,
    o.total_amount,
    o.status,
    o.created_at AS order_date,
    COUNT(oi.id) AS total_items,
    STRING_AGG(p.name || ' (x' || oi.quantity::text || ')', ', ') AS items_list
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
GROUP BY o.id, c.name, o.shipping_address, o.total_amount, o.status, o.created_at;

CREATE OR REPLACE VIEW inventory_status AS
SELECT
    p.id,
    p.name,
    p.inventory_count,
    p.min_stock_level,
    CASE
        WHEN p.inventory_count <= p.min_stock_level THEN 'Low Stock'
        WHEN p.inventory_count <= p.min_stock_level * 2 THEN 'Moderate Stock'
        ELSE 'Good Stock'
    END AS stock_status
FROM products p;

-- Grant necessary permissions (adjust according to your needs)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO current_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO current_user;

-- Verify the seeding
SELECT 'Customers: ' || COUNT(*)::text AS count FROM customers
UNION ALL
SELECT 'Products: ' || COUNT(*)::text FROM products
UNION ALL
SELECT 'Orders: ' || COUNT(*)::text FROM orders
UNION ALL
SELECT 'Order Items: ' || COUNT(*)::text FROM order_items;