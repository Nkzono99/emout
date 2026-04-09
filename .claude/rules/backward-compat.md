<important if="modifying or removing any public API, method, attribute, or class">

# 後方互換ルール

利用者がいる前提で、公開 API の削除・改名・挙動変更は原則行わない。

- `Emout.__getattr__` の動的解決（`p{species}`, `r[eb][xyz]`, `{name}{axis1}{axis2}`）は既存ユーザーコードに直結するため破壊しない。
- 削除が必要な場合は旧名をラッパーとして残し、`warnings.warn(..., DeprecationWarning, stacklevel=2)` → 新実装へ委譲。少なくとも 1 マイナーリリースは共存。削除時期はユーザーに確認。
- 「新メソッド追加 + 既存温存」パターン推奨。例: `Data2d.cmap()`/`contour()` を `plot(mode=...)` と並存させて追加。
- 挙動を変えるときは新キーワード引数（デフォルトは旧挙動）で追加。

</important>
