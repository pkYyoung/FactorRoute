# Databento 数据源

本目录集中存放 Databento 数据源的授权记录、原始下载和适配结果，与其他赛事数据源分开管理。

## 目录约定

- `raw/`：原始下载，保留原始字段与格式。
- `processed/`：清洗、交易日对齐、复权处理后的输入，供 FactorRoute 使用。
- `metadata/`：授权范围、数据字典、下载参数与数据版本记录。

数据文件默认不提交 Git；不要在本目录保存 API Key。后续该数据源的说明、配置和取数脚本统一放在本目录下。

## 如何确认可用范围

1. 登录 Databento Portal，检查 Plans / 数据集详情、Billing / Credits / Data usage 等页面。记录已开通的数据集、历史服务范围、额度与有效期；页面名称可能调整。
2. 查看 Databento 的开通邮件或赛事授权确认，明确学生支持覆盖的数据集、起止日期、schema、下载方式、额度和使用许可。账户可访问或目录中可见，不等于赛事额度覆盖。
3. 如控制台未清楚列出赛事范围，向 Databento 确认上述清单。不要仅凭主办方审核通过推断已获得全部数据权限。
4. 元数据接口 `list_datasets()`、`list_schemas(dataset=...)`、`get_dataset_range(dataset=...)` 可查数据目录、结构和数据集历史覆盖。它们不是个人授权清单。
5. 下载前用 `metadata.get_cost(...)` 估算请求成本；估价不是赛事免费额度确认。明确覆盖后再做小样本下载验证。

建议将确认结果记录到 `metadata/access_scope.md`（本地文件，默认忽略）：数据集、市场、schema、历史起止、额度、到期日、参考数据权限、许可限制、确认来源及确认日期。

## 与现有项目衔接

当前研究框架读取每指标一张 CSV 宽表：第一列日期，其余列证券代码。适配结果写入 `processed/`，再在启动 Python 前设置 `FC_WIDE_CSV_DIR` 指向该目录；文件名需与 `factorcode/project_config.py` 的 `METRIC_FILES` 一致。

在代码、复权、时区、交易日与成交量口径完成验证前，不直接将原始下载当作研究输入。

官方文档：https://databento.com/docs
