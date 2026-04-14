-- HR AI Assistant — PostgreSQL schema and seed data

CREATE TABLE IF NOT EXISTS employees (
    id            SERIAL PRIMARY KEY,
    employee_id   VARCHAR(20) NOT NULL UNIQUE,
    name          VARCHAR(100) NOT NULL,
    email         VARCHAR(150),
    department    VARCHAR(100),
    employment_type VARCHAR(20) DEFAULT 'full-time'  -- full-time | part-time
);

CREATE TABLE IF NOT EXISTS employee_leave (
    id            SERIAL PRIMARY KEY,
    employee_id   VARCHAR(20) NOT NULL REFERENCES employees(employee_id),
    year          INT NOT NULL DEFAULT EXTRACT(YEAR FROM CURRENT_DATE),
    days_taken    INT NOT NULL DEFAULT 0,
    days_pending  INT NOT NULL DEFAULT 0,   -- approved but not yet taken
    UNIQUE (employee_id, year)
);

-- Seed employees
INSERT INTO employees (employee_id, name, email, department, employment_type) VALUES
  ('EMP001', 'John Smith',   'john.smith@company.com',   'Engineering',  'full-time'),
  ('EMP002', 'Sara Jones',   'sara.jones@company.com',   'Marketing',    'full-time'),
  ('EMP003', 'Ahmed Hassan', 'ahmed.hassan@company.com', 'Finance',      'full-time'),
  ('EMP004', 'Lisa Chen',    'lisa.chen@company.com',    'HR',           'part-time')
ON CONFLICT DO NOTHING;

-- Seed leave records for current year
INSERT INTO employee_leave (employee_id, year, days_taken, days_pending) VALUES
  ('EMP001', EXTRACT(YEAR FROM CURRENT_DATE), 17, 2),
  ('EMP002', EXTRACT(YEAR FROM CURRENT_DATE), 8,  0),
  ('EMP003', EXTRACT(YEAR FROM CURRENT_DATE), 22, 0),
  ('EMP004', EXTRACT(YEAR FROM CURRENT_DATE), 5,  1)
ON CONFLICT DO NOTHING;
