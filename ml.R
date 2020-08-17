library("biclust")
a=read.csv("C:\\Users\\msi-\\Desktop\\ml-25m\\mat.csv", header = FALSE)
b=as.matrix(a)
n=400
bc=biclust(b, method=BCBimax(), minr=15, minc=15, number=n)

row = bc@RowxNumber
row = apply(row,2,as.integer)
rowData = row
write.table(rowData, "rowData.txt", row.names = FALSE, col.names = FALSE, sep = ",")

col = bc@NumberxCol
col = apply(col,2,as.integer)
colData = col
write.table(colData, "colData.txt", row.names = FALSE, col.names = FALSE, sep = ",")