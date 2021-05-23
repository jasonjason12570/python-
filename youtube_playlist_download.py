
from pytube import Playlist
import numpy as np
from pytube import Playlist
playlist = Playlist('https://www.youtube.com/playlist?list=PLRfY4Rc-GWzhdCvSPR7aTV0PJjjiSAGMs')
print('Number of videos in playlist: %s' % len(playlist.video_urls))

#get url
url = []
url = list(playlist.video_urls)
url=np.array(url) 
type(url)
url

#get url name
for video in playlist.videos:
    print(video.title)

#download mp3 name
# Loop through all videos in the playlist and download them
for video in playlist.videos:
    video.streams.first().download()
