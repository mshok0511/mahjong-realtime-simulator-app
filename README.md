アプリを起動する際、batファイルを起動するかターミナルから実行コマンドを打ってもらいます。

先ず前提条件として
https://github.com/nekobean/mahjong-cpp/tree/master/python_sample
こちらのGitHubのソースコードをGit Cloneしていただます。
階層は、majan-simulator-appフォルダ内にして、Cloneしたファイルの中にmahjong.pyというpythonファイルがあるはずなので、そちらをapp.pyファイルと同階層に置いてください。
そしたら、mahjong-cpp.zipを同階層に解凍していただきます。

-- batファイルで起動する場合 --
解凍したフォルダの中にsever.exeがあるはずなので、そちらの相対パスをbatファイルに記述してください。
そしたら、batファイルをクリックすると起動します

-- ターミナルからコマンドで実行する場合 --
cd .\majan-simulator-app\
npx electron .

cd 自身のserver.exeファイルの相対パス
server.exe
※sever.exeはコマンドでなくとも、ファイルがある階層まで行き直接実行しても大丈夫です。
