

function mat_to_obj(matfilename, objfilename, isoval)
  % mat_to_obj Convert a .mat file (of a df) to an .obj file of the isosurface at isoval.
  % mat_to_obj(matfilename, objfilename, isoval)
  %     matfilename is the filename the .mat file to process.
  %     objfilename is the filename to save the obj file.
  %     isoval is the value at which to extract the isosurface.
  volume = load(matfilename);
  [faces, vertices] = isosurface(volume.x, isoval);
  if isfield(volume, 'errors') % compute hsv colors for errors
    errs = volume.errors;
    max_err = 3;
    errs(errs > max_err) = max_err;
    if max_err ~= 1, errs = errs / max_err; end
    errs = 1 - errs;
    hsv_h = errs * 240;
    hsv_s = ones(size(errs));
    hsv_v = ones(size(errs)) * 0.5;
    % convert to rgb
    [rgb_r, rgb_g, rgb_b] = ConvertHsvToRgb(hsv_h, hsv_s, hsv_v);
    [colors_x, colors_y, colors_z] = meshgrid(1:size(errs,2), 1:size(errs,1), 1:size(errs,3));
    colors = isocolors(colors_x, colors_y, colors_z, rgb_r, rgb_g, rgb_b, vertices);

    SaveVerticesAndFacesAsObj(vertices, faces, objfilename, colors);
  else
    SaveVerticesAndFacesAsObj(vertices, faces, objfilename);
  end
end


function [r, g, b] = ConvertHsvToRgb(h, s, v)
  % ConvertHsvToRgb Convert hsv colors to rgb colors.
  r = zeros(size(h));
  g = zeros(size(s));
  b = zeros(size(v));

  hd = h / 60;
  hd_floor = floor(hd);
  f = hd - hd_floor;
  p = v .* (1 - s);
  q = v .* (1 - s.*f);
  t = v .* (1 - s.*(1 - f));
  mask = hd_floor == 0;
  r(mask) = v(mask); g(mask) = t(mask); b(mask) = p(mask);
  mask = hd_floor == 6;
  r(mask) = v(mask); g(mask) = t(mask); b(mask) = p(mask);
  mask = hd_floor == 1;
  r(mask) = q(mask); g(mask) = v(mask); b(mask) = p(mask);
  mask = hd_floor == 2;
  r(mask) = p(mask); g(mask) = v(mask); b(mask) = t(mask);
  mask = hd_floor == 3;
  r(mask) = p(mask); g(mask) = q(mask); b(mask) = v(mask);
  mask = hd_floor == 4;
  r(mask) = t(mask); g(mask) = p(mask); b(mask) = v(mask);
  mask = hd_floor == 5;
  r(mask) = v(mask); g(mask) = p(mask); b(mask) = q(mask);
end


function SaveVerticesAndFacesAsObj(v, f, name, vc)
  % SaveVerticesAndFacesAsObj Save a set of vertex coordinates and faces as an .obj file.
  % SaveVerticesAndFacesAsObj(v,f,fname,vc)
  %     v is a Nx3 matrix of vertex coordinates.
  %     f is a Mx3 matrix of vertex indices.
  %     fname is the filename to save the obj file.
  %     vc (optional) is a Nx3 matrix of vertex colors.
  % Check for valid number of function arguments.
  msg = nargchk(3, 4, nargin);
  error(msg)

  fid = fopen(name,'w');

  if nargin == 4
    for i=1:size(v,1)
      fprintf(fid,'v %f %f %f %f %f %f\n',v(i,1),v(i,2),v(i,3),vc(i,1),vc(i,2),vc(i,3));
    end
  else
    for i=1:size(v,1)
      fprintf(fid,'v %f %f %f\n',v(i,1),v(i,2),v(i,3));
    end
  end
  fprintf(fid,'g foo\n');

  for i=1:size(f,1);
    fprintf(fid,'f %d %d %d\n',f(i,1),f(i,2),f(i,3));
  end
  fprintf(fid,'g\n');

  fclose(fid);
end