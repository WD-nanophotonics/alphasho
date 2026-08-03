# 駒画像

`pieces/hitomoji/` の PNG は [Sunfish Shogi Images](https://sunfish-shogi.github.io/shogi-images/)
の `hitomoji` セットです。Web 版では同じファイルを `web/public/pieces/` に複製しています。

インポート時は、アーカイブ直下で
`hitomoji/(black|white)_*.png` に厳密に一致する個別の駒画像だけを採用しました。
path traversal を含む `../docs/...`、合成画像の `piece.png` と
`piece_small.png` は取り込んでいません。

配布元は全画像を [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
でパブリックドメインに提供しており、利用・改変・再配布に許可やクレジットは不要です。
AlphaSho では謝意を込めて配布元へのリンクを Web 版 About 画面にも表示します。
