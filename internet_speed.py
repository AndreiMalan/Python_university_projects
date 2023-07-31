import speedtest

test=speedtest.Speedtest()
download=test.download()
upload=test.upload()
print('Download Speed: {:5.2f} Mb'.format(download/(1024*1024) ))
print('Upload Speed: {:5.2f} Mb'.format(upload/(1024*1024) ))
print(f"Download speed: {download} biti")
print(f"Upload speed: {upload} biti")