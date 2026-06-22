-- create database ott_db;
-- use ott_db;

CREATE TABLE ott_money (
    year INT NOT NULL,
    opid BIGINT NOT NULL,
	svod VARCHAR(100) DEFAULT NULL,
    PRIMARY KEY (year, opid) -- 연도와 유저ID 조합으로 중복 방지
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/ott_money.csv'
INTO TABLE ott_money
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(year, opid, @v_svod)
SET 
    svod = NULLIF(TRIM(@v_svod), ''); 

SELECT 
    u.YEAR, 
    u.OPID AS `고객ID`, 
    u.ott_first, 
    u.ott_second,
    t.`Weekday usage`, 
    t.`Weekend usage`,
    m.svod
FROM ott_usage u
LEFT JOIN ott_time t ON u.OPID = t.OPID AND u.YEAR = t.YEAR
LEFT JOIN ott_money m ON u.OPID = m.OPID AND u.YEAR = m.YEAR