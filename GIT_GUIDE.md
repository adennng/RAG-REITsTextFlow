# Git 使用指南 - 项目开源发布详细步骤

## 🔰 Git 基础概念

### 什么是 Git？
Git 是一个分布式版本控制系统，用于跟踪文件的更改历史，支持多人协作开发。

### 重要概念
- **仓库 (Repository)**: 存储项目文件和历史记录的地方
- **提交 (Commit)**: 保存文件更改的快照
- **分支 (Branch)**: 独立的开发线
- **远程仓库 (Remote)**: 托管在GitHub等平台的仓库

## 📋 开源发布前的准备清单

### ✅ 已完成的准备工作
- [x] 移除敏感信息（密码、API密钥）
- [x] 创建环境变量配置模板 (.env.example)
- [x] 生成依赖包列表 (requirements.txt)
- [x] 编写详细的README.md文档
- [x] 创建.gitignore文件
- [x] 添加MIT开源许可证
- [x] 添加Docker支持文件

## 🚀 Git 操作步骤详解

### 第一步：初始化 Git 仓库

在项目根目录执行：

```bash
# 初始化Git仓库
git init

# 查看当前状态
git status
```

**解释**：
- `git init` 会在当前目录创建一个 `.git` 文件夹，开始跟踪文件变化
- `git status` 显示当前工作区的状态

### 第二步：配置 Git 用户信息

```bash
# 配置用户名（替换为你的名字）
git config --global user.name "你的名字"

# 配置邮箱（替换为你的邮箱）
git config --global user.email "your-email@example.com"

# 查看配置
git config --list
```

**解释**：
- 这些信息会出现在每次提交记录中
- `--global` 表示全局配置，对所有仓库生效

### 第三步：添加文件到暂存区

```bash
# 添加所有文件到暂存区
git add .

# 或者逐个添加文件
git add README.md
git add requirements.txt
git add .env.example

# 查看暂存区状态
git status
```

**解释**：
- `git add .` 添加当前目录下所有文件
- 暂存区是提交前的中间状态
- 绿色文件表示已暂存，红色表示未暂存

### 第四步：创建第一个提交

```bash
# 提交暂存区的所有文件
git commit -m "Initial commit: REITs PDF processing RAG system"

# 查看提交历史
git log --oneline
```

**解释**：
- `-m` 后面是提交信息，简短描述这次更改
- 提交后文件会被永久保存在Git历史中

### 第五步：创建并推送到GitHub

#### 5.1 在GitHub上创建仓库

1. 访问 [GitHub](https://github.com)
2. 点击右上角的 "+" → "New repository"
3. 填写仓库信息：
   - **Repository name**: `rag-Claude` 或 `reits-pdf-rag`
   - **Description**: "中国基础设施公募REITs公告PDF解析RAG系统"
   - **Public**: 选择公开
   - **不要勾选** "Add a README file"（我们已经有了）

#### 5.2 连接远程仓库

```bash
# 添加远程仓库（替换为你的GitHub用户名）
git remote add origin https://github.com/你的用户名/rag-Claude.git

# 查看远程仓库
git remote -v

# 推送到远程仓库
git push -u origin main
```

**解释**：
- `origin` 是远程仓库的默认名称
- `-u` 设置上游分支，之后可以直接用 `git push`

### 第六步：验证发布

访问你的GitHub仓库链接，确认：
- [ ] 所有文件都已上传
- [ ] README.md 显示正常
- [ ] 没有敏感信息泄露

## 🔧 常用 Git 命令

### 日常开发命令

```bash
# 查看文件状态
git status

# 查看文件更改内容
git diff

# 添加特定文件
git add 文件名

# 提交更改
git commit -m "描述信息"

# 推送到远程
git push

# 拉取远程更新
git pull
```

### 分支操作

```bash
# 创建新分支
git branch feature-新功能

# 切换分支
git checkout feature-新功能

# 创建并切换分支（组合命令）
git checkout -b feature-新功能

# 查看所有分支
git branch -a

# 合并分支
git merge feature-新功能

# 删除分支
git branch -d feature-新功能
```

### 查看历史

```bash
# 查看提交历史
git log

# 简洁的历史记录
git log --oneline

# 图形化显示分支
git log --graph --oneline
```

## ⚠️ 重要注意事项

### 1. 敏感信息保护
- **永远不要**提交包含密码、API密钥的文件
- 使用 `.env` 文件存储敏感信息，并在 `.gitignore` 中排除
- 如果意外提交了敏感信息，立即联系有经验的开发者处理

### 2. 提交信息规范
- 使用有意义的提交信息
- 建议格式：`类型: 简短描述`
- 例如：
  - `feat: 添加PDF表格检测功能`
  - `fix: 修复文本提取错误`
  - `docs: 更新README文档`

### 3. 文件管理
- 不要提交大型文件（模型文件、数据文件等）
- 使用 `.gitignore` 排除不必要的文件
- 定期清理临时文件

## 🔄 后续维护流程

### 添加新功能
1. 创建新分支：`git checkout -b feature-新功能名`
2. 开发并测试
3. 提交更改：`git commit -m "添加新功能"`
4. 推送分支：`git push origin feature-新功能名`
5. 在GitHub上创建Pull Request
6. 代码审查后合并

### 修复bug
1. 创建修复分支：`git checkout -b fix-bug描述`
2. 修复问题
3. 提交：`git commit -m "修复bug描述"`
4. 推送并创建Pull Request

### 发布新版本
1. 更新版本号
2. 更新CHANGELOG.md（如果有）
3. 创建发布标签：`git tag v1.0.0`
4. 推送标签：`git push origin v1.0.0`

## 🆘 常见问题解决

### 问题1：推送被拒绝
```bash
# 先拉取远程更新
git pull origin main

# 解决冲突后再推送
git push origin main
```

### 问题2：撤销最后一次提交
```bash
# 撤销提交但保留更改
git reset --soft HEAD~1

# 完全撤销提交和更改（危险！）
git reset --hard HEAD~1
```

### 问题3：查看和恢复历史版本
```bash
# 查看文件的历史版本
git log --follow 文件名

# 恢复文件到特定版本
git checkout 提交哈希值 -- 文件名
```

## 📚 学习资源

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub 使用指南](https://guides.github.com/)
- [廖雪峰的Git教程](https://www.liaoxuefeng.com/wiki/896043488029600)
- [Pro Git 书籍（免费）](https://git-scm.com/book)

## 🎯 下一步建议

1. **设置GitHub Actions**: 自动化测试和部署
2. **添加贡献指南**: 创建 `CONTRIBUTING.md`
3. **设置问题模板**: 便于用户报告bug
4. **添加Wiki文档**: 详细的使用教程
5. **考虑代码质量工具**: 如pre-commit hooks

---

**恭喜！** 你已经完成了项目的开源发布准备。记住，开源是一个持续的过程，需要不断维护和改进。