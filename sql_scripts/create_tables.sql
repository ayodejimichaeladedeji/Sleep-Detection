CREATE TABLE heart_rate (
    id INT IDENTITY(1,1) PRIMARY KEY,
    created_time TIME,
    created_date DATE,
    heart_rate_minute FLOAT
);

CREATE TABLE activity_level_and_calories_burned (
    id INT IDENTITY(1,1) PRIMARY KEY,
    created_time TIME,
    created_date DATE,
    activity_level_minute INT,
    calories_burned_minute FLOAT
);

CREATE TABLE steps (
    id INT IDENTITY(1,1) PRIMARY KEY,
    created_time TIME,
    created_date DATE,
    steps_per_minute INT
);

CREATE TABLE oxygen_saturation (
    id INT IDENTITY(1,1) PRIMARY KEY,
    created_time TIME,
    created_date DATE,
    oxygen_saturation FLOAT
);