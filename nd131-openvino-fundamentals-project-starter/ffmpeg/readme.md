https://superuser.com/questions/1296377/why-am-i-getting-an-unable-to-find-a-suitable-output-format-for-http-localho

-------------------------------------------------------------
just confirmed that ffmpeg Version 4.1.2 does not support the output argument...

http://X.X.X.X:8080/camera.ffm

...when being used for something like...

ffmpeg -nostdin -f rawvideo -pix_fmt bgr32 -s 768x432 -i /data/media/someFileOfYOurs http://X.X.X.X:8080/camera.ffm

...for use with ffserver. This will give the error of...

Unable to find a suitable output format for 'http://X.X.X.X:8080/camera.ffm'
http://X.X.X.X:8080/camera.ffm: Invalid argument

To use ffmpeg along with ffserver you will need to use an older version of ffmpeg.

I have confirmed that ffmpeg Version 3.4.2 does support the output argument referenced above and works with ffserver

---------------------------------------------
How to compile older version that works

ffserver was removed from FFmpeg on 2018-01-06 due to a lack of user and developer interest. 
The last commit including ffserver is 2ca65fc. If you want to use ffserver you can checkout this commit and compile:

git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023
./configure
make -j4




https://askubuntu.com/questions/150906/how-to-install-old-version-of-ffmpeg

sudo apt install ffmpeg=7:3.4.6-0ubuntu0.18.04.1