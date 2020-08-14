# enkf_benchmark

## Description
This is my personal data asimilation benchmark research code.  This includes code for developing and testing data assimilation schemes in the L96-s model.

## Structure
Dynamic model code for L96-s is included in the l96.py file.  Data assimilation methods are included in the ensemble_kalman_schemes.py file.  Drivers are in the benchmark files and submit for slurm scripts are pre-fixed with batch_submit.

## License information

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should be able to access a copy of the license in the following
[License.md](https://github.com/cgrudz/enkf_benchmark/blob/master/LICENSE.md).
If not, see <https://www.gnu.org/licenses/>.
