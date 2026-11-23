# 教室三维模型

通过监控后端打开 `/web/classroom/index.html`。页面支持旋转、平移、滚轮缩放和六个固定机位视角。C6 位于两扇窗之间的空调上方，位置和朝向根据截图估算。浏览器从同一站点读取教室 GLB。

唯一维护的 GLB 位于 `web/classroom/assets/classroom-v3.glb`。它与原教室模型逐字节一致，SHA-256 为 `ed21746db943267fef9dd0d2c9fb51ba6465fc17b9023546f50c19c9e3981912`。当前教室的家具调整、材质、环境光和墙面剖切由本目录的模块绘制。

在项目根目录执行 `node classroom/build.mjs --dependencies "<已安装的 node_modules 绝对路径>"` 即可用现有的 esbuild 和 Three.js 构建。也可在 `classroom` 用 `npm ci` 安装锁定版本，再执行 `npm run build`。默认重新生成 `web/classroom/index.html` 和 Three.js 许可证，原模型留在其发布位置。
