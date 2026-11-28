# 手机内 Android 编译（proot Ubuntu）

无需 Android Studio / Gradle，直接在手机里编 APK。

## 已安装组件

- `default-jdk`（java/javac 21）
- `android-sdk-build-tools`（aapt/aapt2/apksigner/zipalign）
- `android-sdk`（工具集）
- D8：`/opt/r8/r8.jar`（从 Google Maven 下载）
- 平台包：`/usr/lib/android-sdk/platforms/android-29/android.jar`

## 构建

```sh
tools/build-apk.sh /path/to/project
```

项目只需要：

```
AndroidManifest.xml
src/**.java
res/
```

输出 `app-signed.apk`，已签名可安装。

## 限制

- 没有 Gradle/AndroidX；复杂项目仍建议 GitHub Actions
- 构建耗电发热，大项目在手机上很慢
- aapt 用 API 29 平台旧版；新属性（如 android:exported）可能不识别
