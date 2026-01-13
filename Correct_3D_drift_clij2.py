###
# #%L
# Script to register time frames (stacks) to each other using CLIJ2 for translation.
# Ported from Correct_3D_drift.py; drift measurement still uses phase correlation.
# %%
# Copyright (C) 2010 - 2024 Fiji developers.
# %%
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/gpl-3.0.html>.
# #L%
###

# Robert Bryson-Richardson and Albert Cardona 2010-10-08 at Estoril, Portugal
# EMBO Developmental Imaging course by Gabriel Martins
#
# Register time frames (stacks) to each other using Stitching_3D library
# to compute translations only, in all 3 spatial axes.
# Operates on a virtual stack.
# 23/1/13 -
# added user dialog to make use of virtual stack an option
# 10/01/16 -
# Christian Tischer (tischitischer@gmail.com)
# major changes and additions:
# - it now also works for 2D time-series (used to be 3D only)
# - option: measure drift on multiple timescales (this allows to also find slow drift components of less than 1 pixel per frame)
# - option: correct sub-pixel drift computing the shifted images using TransformJ
# - option: if a ROI is put on the image, only this part of the image is considered for drift computation
#           the ROI is moved along with the detected drift thereby tracking the structure of interest
# - macro recording is compatible with previous version
# 21/11/16
# - fixed a bug related to hyperstack conversion
# 06/02/20
# - add field to specify a maximal drift

from ij import VirtualStack, IJ, CompositeImage, ImageStack, ImagePlus
from ij.process import ColorProcessor
from ij.plugin import HyperStackConverter, ZProjector
from ij.io import DirectoryChooser, FileSaver, SaveDialog
from ij.gui import GenericDialog, YesNoCancelDialog, Roi
from mpicbg.imglib.image import ImagePlusAdapter
from mpicbg.imglib.algorithm.fft import PhaseCorrelation
from org.jogamp.vecmath import Point3i
from org.jogamp.vecmath import Point3f
from java.io import File, FilenameFilter
from java.lang import Integer
import math, os, os.path

# CLIJ2
from net.haesleinhuepf.clij2 import CLIJ2
from java.lang import System as JavaSystem
IJ.log("CLIJ2 GPU: " + CLIJ2.getInstance().getGPUName())
from net.imglib2.realtransform import AffineTransform2D, AffineTransform3D

# set native DLL path if build output is present and env var not set
_vkfft_env = os.environ.get("CLIJX_VKFFT_PATH")
_script_dir = None
try:
  _script_dir = os.path.dirname(os.path.abspath(__file__))
except Exception:
  _script_dir = os.getcwd()
_vkfft_default = os.path.join(_script_dir, "clijx", "src", "main", "native",
                              "vkfft_phase_correlation", "build", "clijx_vkfft.dll")
if (_vkfft_env is None or _vkfft_env == "") and os.path.exists(_vkfft_default):
  os.environ["CLIJX_VKFFT_PATH"] = _vkfft_default
  JavaSystem.setProperty("CLIJX_VKFFT_PATH", _vkfft_default)
  IJ.log("CLIJX_VKFFT_PATH set to " + _vkfft_default)
elif (_vkfft_env is None or _vkfft_env == ""):
  IJ.log("CLIJX_VKFFT_PATH not set and default DLL not found at " + _vkfft_default)
else:
  JavaSystem.setProperty("CLIJX_VKFFT_PATH", _vkfft_env)

from net.haesleinhuepf.clijx.plugins import PhaseCorrelationFFT

# sub-pixel translation using imglib2 (CPU fallback)
from net.imagej.axis import Axes
from net.imglib2.img.display.imagej import ImageJFunctions
from net.imglib2.realtransform import RealViews, Translation3D, Translation2D
from net.imglib2.view import Views
from net.imglib2.img.imageplus import ImagePlusImgs
from net.imglib2.converter import Converters
from net.imglib2.converter.readwrite import RealFloatSamplerConverter
from net.imglib2.interpolation.randomaccess import NLinearInterpolatorFactory

def translate_single_stack_using_imglib2(imp, dx, dy, dz):
  # wrap into a float imglib2 and translate
  #   conversion into float is necessary due to "overflow of n-linear interpolation due to accuracy limits of unsigned bytes"
  #   see: https://github.com/fiji/fiji/issues/136#issuecomment-173831951
  img = ImagePlusImgs.from(imp.duplicate())
  extended = Views.extendZero(img)
  converted = Converters.convert(extended, RealFloatSamplerConverter())
  interpolant = Views.interpolate(converted, NLinearInterpolatorFactory())

  # translate
  if imp.getNDimensions()==3:
    transformed = RealViews.affine(interpolant, Translation3D(dx, dy, dz))
  elif imp.getNDimensions()==2:
    transformed = RealViews.affine(interpolant, Translation2D(dx, dy))
  else:
    IJ.log("Can only work on 2D or 3D stacks")
    return None

  cropped = Views.interval(transformed, img)
  # wrap back into bit depth of input image and return
  bd = imp.getBitDepth()
  if bd==8:
    return(ImageJFunctions.wrapUnsignedByte(cropped,"imglib2"))
  elif bd == 16:
    return(ImageJFunctions.wrapUnsignedShort(cropped,"imglib2"))
  elif bd == 32:
    return(ImageJFunctions.wrapFloat(cropped,"imglib2"))
  else:
    return None


def gpu_translate_stack(imp, dx, dy, dz, subpixel):
  clij2 = CLIJ2.getInstance()
  src = clij2.push(imp)
  dst = clij2.create(src)

  if imp.getNSlices() <= 1:
    if subpixel:
      at = AffineTransform2D()
      at.translate(dx, dy)
      clij2.affineTransform2D(src, dst, at)
    else:
      clij2.translate2D(src, dst, float(dx), float(dy))
  else:
    if subpixel:
      at = AffineTransform3D()
      at.translate(dx, dy, dz)
      clij2.affineTransform3D(src, dst, at)
    else:
      clij2.translate3D(src, dst, float(dx), float(dy), float(dz))

  result = clij2.pull(dst)
  src.close()
  dst.close()
  return result


def compute_shift_cpu(imp1, imp2):
  """ Compute a Point3i that expressed the translation of imp2 relative to imp1 (CPU phase correlation)."""
  phc = PhaseCorrelation(ImagePlusAdapter.wrap(imp1), ImagePlusAdapter.wrap(imp2), 5, True)
  phc.process()
  p = phc.getShift().getPosition()
  if len(p)==3: # 3D data
    p3 = p
  elif len(p)==2: # 2D data: add zero shift for z
    p3 = [p[0],p[1],0]
  return Point3i(p3)


def compute_shift(imp1, imp2):
  """ Compute a Point3i that expressed the translation of imp2 relative to imp1 using GPU phase correlation if available."""
  try:
    clij2 = CLIJ2.getInstance()
    buf1 = clij2.push(imp1)
    buf2 = clij2.push(imp2)
    shift = PhaseCorrelationFFT.phaseCorrelationShift(clij2, buf1, buf2)
    buf1.close()
    buf2.close()
    IJ.log("GPU phase correlation active")
    return Point3i(int(round(shift[0])), int(round(shift[1])), int(round(shift[2])))
  except Exception as e:
    IJ.log("GPU phase correlation unavailable, falling back to CPU: " + str(e))
    return compute_shift_cpu(imp1, imp2)


def extract_frame(imp, frame, channel, z_min, z_max):
  """ From a VirtualStack that is a hyperstack, contained in imp,
  extract the timepoint frame as an ImageStack, and return it.
  It will do so only for the given channel. """
  stack = imp.getStack() # multi-time point virtual stack
  stack2 = ImageStack(imp.width, imp.height, None)
  for s in range(int(z_min), int(z_max)+1):
    i = imp.getStackIndex(channel, s, frame)
    stack2.addSlice(str(s), stack.getProcessor(i))
  return stack2


def extract_frame_process_roi(imp, frame, roi, options):
  # extract frame and channel
  imp_frame = ImagePlus("", extract_frame(imp, frame, options['channel'], options['z_min'], options['z_max'])).duplicate()
  # check for roi and crop
  if roi != None:
    imp_frame.setRoi(roi)
    IJ.run(imp_frame, "Crop", "")
  # subtract background
  if options['background'] > 0:
    IJ.run(imp_frame, "Subtract...", "value="+str(options['background'])+" stack")
  # enhance edges
  if options['process']:
    IJ.run(imp_frame, "Mean 3D...", "x=1 y=1 z=0")
    IJ.run(imp_frame, "Find Edges", "stack")
  # project into 2D if we only want to correct the drift in x and y
  if imp_frame.getNSlices() > 1:
    if options['correct_only_xy']:
      imp_frame = ZProjector.run(imp_frame, "avg")
  # return
  return imp_frame


def add_Point3f(p1, p2):
  p3 = Point3f(0,0,0)
  p3.x = p1.x + p2.x
  p3.y = p1.y + p2.y
  p3.z = p1.z + p2.z
  return p3


def subtract_Point3f(p1, p2):
  p3 = Point3f(0,0,0)
  p3.x = p1.x - p2.x
  p3.y = p1.y - p2.y
  p3.z = p1.z - p2.z
  return p3


def get_Point3i(point, dimension):
  if dimension == 0:
    return point.x
  if dimension == 1:
    return point.y
  if dimension == 2:
    return point.z
  else:
    IJ.log("Tried to get Point3f at coordinate " + str( dimension ))


def set_Point3i(point, dimension, value):
  if dimension == 0:
    point.x = int(value)
    return
  if dimension == 1:
    point.y = int(value)
    return
  if dimension == 2:
    point.z = int(value)
    return
  else:
    IJ.log("Tried to set Point3f at coordinate " + str( dimension ))


def shift_between_rois(roi2, roi1):
  """ computes the relative xy shift between two rois
  """
  dr = Point3f(0,0,0)
  dr.x = roi2.getBounds().x - roi1.getBounds().x
  dr.y = roi2.getBounds().y - roi1.getBounds().y
  dr.z = 0
  return dr


def shift_roi(imp, roi, dr):
  """ shifts a roi in x,y by dr.x and dr.y
  if the shift would cause the roi to be outside the imp,
  it only shifts as much as possible maintaining the width and height
  of the input roi
  """
  if roi == None:
    return roi
  else:
    r = roi.getBounds()
    # init x,y coordinates of new shifted roi
    sx = 0
    sy = 0
    # x shift
    if (r.x + dr.x) < 0:
      sx = 0
    elif (r.x + dr.x + r.width) > imp.width:
      sx = int(imp.width-r.width)
    else:
      sx = r.x + int(dr.x)
    # y shift
    if (r.y + dr.y) < 0:
      sy = 0
    elif (r.y + dr.y + r.height) > imp.height:
      sy = int(imp.height-r.height)
    else:
      sy = r.y + int(dr.y)
    # return shifted roi
    shifted_roi = Roi(sx, sy, r.width, r.height)
    return shifted_roi


def compute_and_update_frame_translations_dt(imp, dt, options, shifts = None):
  """ imp contains a hyper virtual stack, and we want to compute
  the X,Y,Z translation between every t and t+dt time points in it
  using the given preferred channel.
  if shifts were already determined at other (lower) dt
  they will be used and updated.
  """
  nt = imp.getNFrames()
  # get roi (could be None)
  roi = imp.getRoi()
  # init shifts
  if shifts == None:
    shifts = []
    for t in range(nt):
      shifts.append(Point3f(0,0,0))
  # compute shifts
  IJ.showProgress(0)
  max_shifts = options['max_shifts']
  for t in range(dt, nt+dt, dt):
    if t > nt-1: # together with above range till nt+dt this ensures that the last data points are not missed out
      t = nt-1 # nt-1 is the last shift (0-based)
    IJ.log("      between frames "+str(t-dt+1)+" and "+str(t+1))
    # get (cropped and processed) image at t-dt
    roi1 = shift_roi(imp, roi, shifts[t-dt])
    imp1 = extract_frame_process_roi(imp, t+1-dt, roi1, options)
    # get (cropped and processed) image at t
    roi2 = shift_roi(imp, roi, shifts[t])
    imp2 = extract_frame_process_roi(imp, t+1, roi2, options)
    # compute shift
    local_new_shift = compute_shift(imp2, imp1)
    limit_shifts_to_maximal_shifts(local_new_shift, max_shifts)

    if roi: # total shift is shift of rois plus measured drift
      local_new_shift = add_Point3f(local_new_shift, shift_between_rois(roi2, roi1))
    # determine the shift that we knew already
    local_shift = subtract_Point3f(shifts[t],shifts[t-dt])
    # compute difference between new and old measurement (which come from different dt)
    add_shift = subtract_Point3f(local_new_shift,local_shift)
    # update shifts from t-dt to the end (assuming that the measured local shift will persist till the end)
    for i,tt in enumerate(range(t-dt,nt)):
      # for i>dt below expression basically is a linear drift prediction for the frames at tt>t
      # this is only important for predicting the best shift of the ROI
      # the drifts for i>dt will be corrected by the next measurements
      shifts[tt].x += 1.0*i/dt * add_shift.x
      shifts[tt].y += 1.0*i/dt * add_shift.y
      shifts[tt].z += 1.0*i/dt * add_shift.z
    IJ.showProgress(1.0*t/(nt+1))

  IJ.showProgress(1)
  return shifts


def limit_shifts_to_maximal_shifts(local_new_shift, max_shifts):
  for d in range(3):
    shift = get_Point3i(local_new_shift, d)
    if shift > max_shifts[d]:
      IJ.log("Too large drift along dimension " + str(d)
         + ":  " + str(shift)
         + "; restricting to " + str(int(max_shifts[d])))
      set_Point3i(local_new_shift, d, int(max_shifts[d]))
      continue
    if shift < -1 * max_shifts[d]:
      IJ.log("Too large drift along dimension " + str(d)
         + ":  " + str(shift)
         + "; restricting to " + str(int(-1 * max_shifts[d])))
      set_Point3i(local_new_shift, d, int(-1 * max_shifts[d]))
      continue


def convert_shifts_to_integer(shifts):
  int_shifts = []
  for shift in shifts:
    int_shifts.append(Point3i(int(round(shift.x)),int(round(shift.y)),int(round(shift.z))))
  return int_shifts


def compute_min_max(shifts):
  """ Find out the top left up corner, and the right bottom down corner,
  namely the bounds of the new virtual stack to create.
  Expects absolute shifts. """
  minx = Integer.MAX_VALUE
  miny = Integer.MAX_VALUE
  minz = Integer.MAX_VALUE
  maxx = -Integer.MAX_VALUE
  maxy = -Integer.MAX_VALUE
  maxz = -Integer.MAX_VALUE
  for shift in shifts:
    minx = min(minx, shift.x)
    miny = min(miny, shift.y)
    minz = min(minz, shift.z)
    maxx = max(maxx, shift.x)
    maxy = max(maxy, shift.y)
    maxz = max(maxz, shift.z)
  return minx, miny, minz, maxx, maxy, maxz


def zero_pad(num, digits):
  """ for 34, 4 --> '0034' """
  str_num = str(num)
  while (len(str_num) < digits):
    str_num = '0' + str_num
  return str_num


def invert_shifts(shifts):
  """ invert shifts such that they can be used for correction.
  """
  for shift in shifts:
    shift.x *= -1
    shift.y *= -1
    shift.z *= -1
  return shifts


def register_hyperstack_clij2(imp, shifts, target_folder, virtual, subpixel):
  """ Applies the shifts to all channels in the hyperstack using CLIJ2 for translation."""
  # Compute bounds of the new volume,
  # which accounts for all translations:
  minx, miny, minz, maxx, maxy, maxz = compute_min_max(shifts)
  # Make shifts relative to new canvas dimensions
  # so that the min values become 0,0,0
  for shift in shifts:
    shift.x -= minx
    shift.y -= miny
    shift.z -= minz
  # new canvas dimensions:
  width = int(imp.width + maxx - minx)
  height = int(maxy - miny + imp.height)
  slices = int(maxz - minz + imp.getNSlices())

  empty = imp.getProcessor().createProcessor(width, height)
  if isinstance(empty, ColorProcessor):
    empty.setValue(0)
    empty.fill()

  if virtual is True:
    names = []
  else:
    registeredstack = ImageStack(width, height, imp.getProcessor().getColorModel())

  stack = imp.getStack()
  IJ.showProgress(0)

  for frame in range(1, imp.getNFrames()+1):
    IJ.showProgress(frame / float(imp.getNFrames()+1))
    fr = "t" + zero_pad(frame, len(str(imp.getNFrames())))

    shift = shifts[frame-1]
    IJ.log("    frame "+str(frame)+" correcting drift "+str(round(-shift.x-minx,2))+","+str(round(-shift.y-miny,2))+","+str(round(-shift.z-minz,2)))

    for ch in range(1, imp.getNChannels()+1):
      tmpstack = ImageStack(width, height, imp.getProcessor().getColorModel())

      # get all slices of this channel and frame
      for s in range(1, imp.getNSlices()+1):
        ip = stack.getProcessor(imp.getStackIndex(ch, s, frame))
        ip2 = ip.createProcessor(width, height)
        ip2.insert(ip, 0, 0)
        tmpstack.addSlice("", ip2)

      # pad the end (in z) of this channel and frame
      for s in range(imp.getNSlices(), slices):
        tmpstack.addSlice("", empty)

      imp_tmpstack = ImagePlus("", tmpstack)
      if subpixel:
        imp_translated = gpu_translate_stack(imp_tmpstack, shift.x, shift.y, shift.z, True)
      else:
        imp_translated = gpu_translate_stack(imp_tmpstack, shift.x, shift.y, shift.z, False)

      translated_stack = imp_translated.getStack()
      for s in range(1, translated_stack.getSize()+1):
        ss = "_z" + zero_pad(s, len(str(slices)))
        ip = translated_stack.getProcessor(s).duplicate()
        if virtual is True:
          name = fr + ss + "_c" + zero_pad(ch, len(str(imp.getNChannels()))) +".tif"
          names.append(name)
          currentslice = ImagePlus("", ip)
          currentslice.setCalibration(imp.getCalibration().copy())
          currentslice.setProperty("Info", imp.getProperty("Info"))
          FileSaver(currentslice).saveAsTiff(target_folder + "/" + name)
        else:
          registeredstack.addSlice("", ip)

  IJ.showProgress(1)

  if virtual is True:
    registeredstack = VirtualStack(width, height, None, target_folder)
    for name in names:
      registeredstack.addSlice(name)

  registeredstack_imp = ImagePlus("registered time points", registeredstack)
  registeredstack_imp.setCalibration(imp.getCalibration().copy())
  registeredstack_imp.setProperty("Info", imp.getProperty("Info"))
  registeredstack_imp = HyperStackConverter.toHyperStack(registeredstack_imp, imp.getNChannels(), slices, imp.getNFrames(), "xyzct", "Composite")

  return registeredstack_imp


class Filter(FilenameFilter):
  def accept(self, folder, name):
    return not File(folder.getAbsolutePath() + "/" + name).isHidden()


def validate(target_folder):
  f = File(target_folder)
  if len(File(target_folder).list(Filter())) > 0:
    yn = YesNoCancelDialog(IJ.getInstance(), "Warning!", "Target folder is not empty! May overwrite files! Continue?")
    if yn.yesPressed():
      return True
    else:
      return False
  return True


def getOptions(imp):
  gd = GenericDialog("Correct 2D/3D Drift Options")
  channels = []
  for ch in range(1, imp.getNChannels()+1 ):
    channels.append(str(ch))
  gd.addChoice("Channel for registration:", channels, channels[0])
  gd.addCheckbox("Correct only x & y (for 3D data):", False)
  gd.addCheckbox("Multi_time_scale computation for enhanced detection of slow drifts?", False)
  gd.addCheckbox("Sub_pixel drift correction (possibly needed for slow drifts)?", False)
  gd.addCheckbox("Edge_enhance images for possibly improved drift detection?", False)
  gd.addNumericField("Only consider pixels with values larger than:", 0, 0)
  gd.addNumericField("Lowest z plane to take into account:", 1, 0)
  gd.addNumericField("Highest z plane to take into account:", imp.getNSlices(), 0)
  gd.addNumericField("Max_shift_x [pixels]:", 10, imp.getWidth())
  gd.addNumericField("Max_shift_y [pixels]:", 10, imp.getHeight())
  gd.addNumericField("Max_shift_z [pixels]:", 10, imp.getNSlices())
  gd.addCheckbox("Use virtualstack for saving the results to disk to save RAM?", False)
  gd.addCheckbox("Only compute drift vectors?", False)
  gd.addMessage("If you put a ROI, drift will only be computed in this region;\n the ROI will be moved along with the drift to follow your structure of interest.")
  gd.showDialog()
  if gd.wasCanceled():
    return
  options = {}
  options['channel'] = gd.getNextChoiceIndex() + 1  # zero-based
  options['correct_only_xy'] = gd.getNextBoolean()
  options['multi_time_scale'] = gd.getNextBoolean()
  options['subpixel'] = gd.getNextBoolean()
  options['process'] = gd.getNextBoolean()
  options['background'] = gd.getNextNumber()
  options['z_min'] = gd.getNextNumber()
  options['z_max'] = gd.getNextNumber()
  max_shifts = [0,0,0]
  max_shifts[0] = gd.getNextNumber()
  max_shifts[1] = gd.getNextNumber()
  max_shifts[2] = gd.getNextNumber()
  options['max_shifts'] = max_shifts
  options['virtual'] = gd.getNextBoolean()
  options['only_compute'] = gd.getNextBoolean()
  return options


def save_shifts(shifts, roi):
  sd = SaveDialog('please select shift file for saving', 'shifts', '.txt')
  fp = os.path.join(sd.getDirectory(),sd.getFileName())
  f = open(fp, 'w')
  txt = []
  txt.append("ROI zero-based")
  txt.append("\nx_min\ty_min\tz_min\tx_max\ty_max\tz_max")
  txt.append("\n"+str(roi[0])+"\t"+str(roi[1])+"\t"+str(roi[2])+"\t"+str(roi[3])+"\t"+str(roi[4])+"\t"+str(roi[5]))
  txt.append("\nShifts")
  txt.append("\ndx\tdy\tdz")
  for shift in shifts:
    txt.append("\n"+str(shift.x)+"\t"+str(shift.y)+"\t"+str(shift.z))
  f.writelines(txt)
  f.close()


def run():

  IJ.log("Correct_3D_Drift (CLIJ2 translation)")

  imp = IJ.getImage()
  if imp is None:
    return
  if 1 == imp.getNFrames():
    IJ.showMessage("Cannot register because there is only one time frame.\nPlease check [Image > Properties...].")
    return

  options = getOptions(imp)
  if options is None:
    return # user pressed Cancel

  if options['z_min'] < 1:
    IJ.showMessage("The minimal z plane must be >=1.")
    return

  if options['z_max'] > imp.getNSlices():
    IJ.showMessage("Your image only has "+str(imp.getNSlices())+" z-planes, please adapt your z-range.")
    return

  if options['virtual'] is True:
    dc = DirectoryChooser("Choose target folder to save image sequence")
    target_folder = dc.getDirectory()
    if target_folder is None:
      return # user canceled the dialog
    if not validate(target_folder):
      return
  else:
    target_folder = None

  #
  # compute drift
  #
  IJ.log("  computing drift...")

  IJ.log("    at frame shifts of 1")
  dt = 1; shifts = compute_and_update_frame_translations_dt(imp, dt, options)

  # multi-time-scale computation
  if options['multi_time_scale'] is True:
    dt_max = imp.getNFrames()-1
    # computing drifts on exponentially increasing time scales 3^i up to 3^6
    dts = [3,9,27,81,243,729,dt_max]
    for dt in dts:
      if dt < dt_max:
        IJ.log("    at frame shifts of "+str(dt))
        shifts = compute_and_update_frame_translations_dt(imp, dt, options, shifts)
      else:
        IJ.log("    at frame shifts of "+str(dt_max))
        shifts = compute_and_update_frame_translations_dt(imp, dt_max, options, shifts)
        break

  # invert measured shifts to make them the correction
  shifts = invert_shifts(shifts)

  # apply shifts
  if not options['only_compute']:

    IJ.log("  applying shifts...")

    if not options['subpixel']:
      shifts = convert_shifts_to_integer(shifts)

    registered_imp = register_hyperstack_clij2(imp, shifts, target_folder, options['virtual'], options['subpixel'])

    if options['virtual'] is True:
      if 1 == imp.getNChannels():
        ip=imp.getProcessor()
        ip2=registered_imp.getProcessor()
        ip2.setColorModel(ip.getCurrentColorModel())
      else:
        registered_imp.copyLuts(imp)
    else:
      if imp.getNChannels() > 1:
        registered_imp.copyLuts(imp)

    registered_imp.show()

  else:

    if imp.getRoi():
      xmin = imp.getRoi().getBounds().x
      ymin = imp.getRoi().getBounds().y
      zmin = 0
      xmax = xmin + imp.getRoi().getBounds().width - 1
      ymax = ymin + imp.getRoi().getBounds().height - 1
      zmax = imp.getNSlices()-1
    else:
      xmin = 0
      ymin = 0
      zmin = 0
      xmax = imp.getWidth() - 1
      ymax = imp.getHeight() - 1
      zmax = imp.getNSlices() - 1

    save_shifts(shifts, [xmin, ymin, zmin, xmax, ymax, zmax])
    IJ.log("  saving shifts...")

run()
