require 'nn'
require 'optim'
require 'nngraph'

-- split string into table (source stackoverflow)
function split(str, sep)
	sep = sep or ','
	fields={}
	local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
	if not matchfunc then return {str} end
	for str in matchfunc do
		table.insert(fields, str)
	end
	return fields
end

-- read csv into table (source stackoverflow)
function read(path, sep, tonum)
	tonum = tonum or true
	sep = sep or ','
	local csvFile = {}
	local file = assert(io.open(path, "r"))
	for line in file:lines() do
		fields = split(line, sep)
		if tonum then -- convert numeric fields to numbers
			for i=1,#fields do
				fields[i] = tonumber(fields[i]) or fields[i]
			end
		end
		table.insert(csvFile, fields)
	end
	file:close()
	return csvFile
end