from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, clips_array, vfx

def create_clip(ride):
    clip = VideoFileClip(ride + ".mp4").margin(5)
    txt_clip = TextClip('Track 1, dataset ' + ride)
    txt_clip = txt_clip.set_position((10, 10)).set_duration(160)
    return CompositeVideoClip([clip, txt_clip])

clips = ['01', '02', '04', '05', '0102', '0405', '0204', '0205', '010204', '010205', '020405', '01020405']

max_row_count = 3
clip_array = []
clip_row = []

for clip in clips:
   clip_row.append(create_clip(clip))
   if len(clip_row) >= max_row_count:
       clip_array.append(clip_row)
       clip_row = []

final_clip = clips_array(clip_array)
# final_clip.resize(width=480).write_videofile("all_combined.mp4")
final_clip.write_videofile("all_combined.mp4")
