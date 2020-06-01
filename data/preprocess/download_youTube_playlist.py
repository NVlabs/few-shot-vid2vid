import glob
from pytube import YouTube, Playlist


# Example script to download youtube scripts.
class MyPlaylist(Playlist):
    def download_all(
        self,
        download_path=None,
        prefix_number=True,
        reverse_numbering=False,
        idx=0,
    ):                    
        self.populate_video_urls()
        prefix_gen = self._path_num_prefix_generator(reverse_numbering)        
        for i, link in enumerate(self.video_urls):            
            prefix = '%03d_%03d_' % (idx + 1, i + 1)
            p = glob.glob(prefix[:-1] + '*.mp4')
            print(prefix, link)
            if not p:
                try:                    
                    yt = YouTube(link)                
                    dl_stream = yt.streams.filter(adaptive=True, subtype='mp4').first()                    
                    dl_stream.download(download_path, filename_prefix=prefix)
                except:
                    print('cannot download')
                    pass

playlist_path = 'youTube_playlist.txt'            
with open(playlist_path, 'r') as f:
    playlists = f.read().splitlines()

for i, playlist in enumerate(playlists):
    pl = MyPlaylist(playlist)
    pl.download_all(idx=i)
