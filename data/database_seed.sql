CREATE TABLE IF NOT EXISTS support_articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    team TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO support_articles (id, title, body, team, updated_at) VALUES
    (
        'shipping-101',
        'Domestic shipping time',
        'After an order leaves the warehouse, standard domestic delivery usually takes 3 to 5 business days. A tracking link is emailed when the package is dispatched.',
        'fulfillment',
        '2026-01-15'
    ),
    (
        'orders-204',
        'Changing an order',
        'Support can update a shipping address before an order is dispatched. Once a package has shipped, the customer should use the carrier tracking link to check whether delivery options are available.',
        'support',
        '2026-02-03'
    ),
    (
        'claims-305',
        'Warranty claim review',
        'The support team usually reviews a complete warranty claim within 5 business days. If more information is needed, support contacts the customer using the email address on the order.',
        'support',
        '2026-02-18'
    ),
    (
        'AWARDS-1',
        'Noble prize winner',
        'Harish Kunta is Winner of noble price',
        'AWARDS',
        '2026-03-18'
    );
