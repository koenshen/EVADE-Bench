
WITH ground_truth AS (
    SELECT
        id,
        model,
        from_json(regexp_replace(选项, "'", '"'),'array<string>') AS ground_truth_options
    FROM
        table_name
),
other_models AS (
    SELECT
        id,
        model,
        from_json(regexp_replace(选项, "'", '"'), 'array<string>') AS other_options
    FROM
        table_name
),
comparison AS (
    -- 合并 ground truth 和其他模型数据，计算交集和相等性
    SELECT
        o.id,
        o.model,
        g.ground_truth_options,
        o.other_options,
        -- 判断是否有交集
        CASE
            WHEN SIZE(ARRAY_INTERSECT(o.other_options, g.ground_truth_options)) > 0 THEN 1
            ELSE 0
        END AS has_intersection,
        -- 判断是否完全相等
        CASE
            WHEN
                size(array_except(o.other_options, g.ground_truth_options)) = 0
                AND
                size(array_except(g.ground_truth_options, o.other_options)) = 0
            THEN 1
            ELSE 0
        END AS is_equal
    FROM
        other_models o
    JOIN
        ground_truth g
    ON
        o.id = g.id
),
accuracy_calculation AS (
    -- 按 id 和 model 分组，计算部分正确率和完全正确率
    SELECT
        id,
        model,
        -- 部分正确率
        AVG(has_intersection) AS partial_accuracy,
        -- 完全正确率
        AVG(is_equal) AS full_accuracy
    FROM
        comparison
    GROUP BY
        id, model
),
final_aggregation AS (
    -- 按 model 分组，计算所有样本的平均准确率
    SELECT
        model,
        AVG(partial_accuracy) AS average_partial_accuracy,
        AVG(full_accuracy) AS average_full_accuracy
    FROM
        accuracy_calculation
    GROUP BY
        model
)
SELECT
    model,
    average_partial_accuracy AS partial_accuracy,
    average_full_accuracy AS full_accuracy
FROM
    final_aggregation
ORDER BY partial_accuracy desc;
